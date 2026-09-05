#include "gpt2/model.h"

#include "gpt2/layers.h"
#include "gpt2/tensor_ops.h"
#include "gpt2/transformer.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gpt2 {

namespace {

constexpr std::size_t top_level_tensor_count = 4;
constexpr std::size_t tensors_per_layer = 12;

// A quantized tensor costs one extra checkpoint record (its
// ".quant_scale" companion), so the true worst-case totals below --
// used only to keep the overflow guard in validate_checkpoint_schema
// correct, not as the expected count itself -- add one for wte and
// one per per-layer quantizable weight (4).
constexpr std::size_t max_top_level_tensor_count =
    top_level_tensor_count + 1;
constexpr std::size_t max_tensors_per_layer = tensors_per_layer + 4;

std::uint64_t next_model_identity() {
    static std::atomic<std::uint64_t> next{1};
    return next.fetch_add(1, std::memory_order_relaxed);
}

std::size_t checked_product(
    std::size_t value,
    std::size_t multiplier,
    const char* description
) {
    if (value > std::numeric_limits<std::size_t>::max() / multiplier) {
        throw std::invalid_argument(description);
    }

    return value * multiplier;
}

const Tensor& require_tensor(
    const Checkpoint& checkpoint,
    const std::string& name,
    const Tensor::Shape& expected_shape
) {
    if (!checkpoint.contains(name)) {
        throw std::invalid_argument(
            "checkpoint is missing required tensor: " + name
        );
    }

    const Tensor& tensor = checkpoint.tensor(name);
    if (tensor.shape() != expected_shape) {
        throw std::invalid_argument(
            "checkpoint tensor has an invalid shape: " + name
        );
    }

    return tensor;
}

std::string layer_prefix(std::size_t layer) {
    return "transformer.h." + std::to_string(layer);
}

enum class TensorPrecision {
    fp32,
    int8,
};

// Like require_tensor, but `name` may instead be a per-channel-
// quantized int8 tensor: symmetric per-channel quantization (see
// docs/quantization.md) with an accompanying ".quant_scale" FP32
// tensor. `scale_axis` is which of expected_shape's dimensions is the
// channel axis for this specific tensor — that is a property of what
// the tensor represents, not something a shape can answer on its own
// (see the note on c_proj's square weight matrix in
// docs/quantization.md), so every call site below states it
// explicitly. Returns which of the two forms was found.
TensorPrecision require_linear_weight(
    const Checkpoint& checkpoint,
    const std::string& name,
    const Tensor::Shape& expected_shape,
    std::size_t scale_axis
) {
    if (checkpoint.contains_int8(name)) {
        const Int8Tensor& tensor = checkpoint.int8_tensor(name);
        if (tensor.shape() != expected_shape) {
            throw std::invalid_argument(
                "checkpoint tensor has an invalid shape: " + name
            );
        }

        const std::string scale_name = name + ".quant_scale";
        if (!checkpoint.contains(scale_name)) {
            throw std::invalid_argument(
                "checkpoint is missing the quantization scale for: " +
                name
            );
        }

        const Tensor& scale = checkpoint.tensor(scale_name);
        if (scale.rank() != 1 ||
            scale.numel() != expected_shape.at(scale_axis)) {
            throw std::invalid_argument(
                "checkpoint quantization scale has an invalid shape: " +
                name
            );
        }

        return TensorPrecision::int8;
    }

    static_cast<void>(require_tensor(checkpoint, name, expected_shape));
    return TensorPrecision::fp32;
}

void validate_checkpoint_schema(const Checkpoint& checkpoint) {
    const ModelConfig& config = checkpoint.config();
    const std::size_t vocabulary_size =
        static_cast<std::size_t>(config.vocab_size);
    const std::size_t context_length =
        static_cast<std::size_t>(config.context_length);
    const std::size_t embedding_size =
        static_cast<std::size_t>(config.embedding_size);
    const std::size_t head_count =
        static_cast<std::size_t>(config.head_count);
    const std::size_t layer_count =
        static_cast<std::size_t>(config.layer_count);

    if (embedding_size % head_count != 0) {
        throw std::invalid_argument(
            "checkpoint embedding size must be divisible by head count"
        );
    }

    if (layer_count >
        (std::numeric_limits<std::size_t>::max() -
         max_top_level_tensor_count) / max_tensors_per_layer) {
        throw std::invalid_argument(
            "checkpoint layer count is too large"
        );
    }

    const std::size_t qkv_size = checked_product(
        embedding_size,
        3,
        "checkpoint embedding size is too large for QKV parameters"
    );
    const std::size_t feed_forward_size = checked_product(
        embedding_size,
        4,
        "checkpoint embedding size is too large for feed-forward parameters"
    );

    // A quantized tensor occupies two checkpoint records (its int8
    // payload and its ".quant_scale" companion) rather than one, so
    // the exact record count the schema expects depends on how many
    // of the five quantizable tensors turn out to be int8; that is
    // only known once every one of them has been inspected below, so
    // the total-count check itself happens at the end of this
    // function rather than up front.
    std::size_t quantized_tensor_count = 0;

    // wte's channel axis is its rows (axis 0) -- one per vocabulary
    // word -- not its columns; see docs/quantization.md.
    if (require_linear_weight(
            checkpoint,
            "transformer.wte.weight",
            {vocabulary_size, embedding_size},
            0
        ) == TensorPrecision::int8) {
        ++quantized_tensor_count;
    }
    static_cast<void>(require_tensor(
        checkpoint,
        "transformer.wpe.weight",
        {context_length, embedding_size}
    ));

    // Every layer's four linear weights are quantized together or not
    // at all: transformer_block/quantized_transformer_block each bind
    // a whole block to one precision, so a layer mixing the two has
    // no block function to run it. tools/export_gpt2.py's --quantize
    // flag likewise applies to every transformer layer uniformly, so a
    // checkpoint where layers disagree cannot come from that exporter
    // and is rejected here rather than silently accepted.
    bool transformer_weight_precision_known = false;
    TensorPrecision transformer_weight_precision = TensorPrecision::fp32;

    for (std::size_t layer = 0; layer < layer_count; ++layer) {
        const std::string prefix = layer_prefix(layer);

        static_cast<void>(require_tensor(
            checkpoint,
            prefix + ".ln_1.weight",
            {embedding_size}
        ));
        static_cast<void>(require_tensor(
            checkpoint,
            prefix + ".ln_1.bias",
            {embedding_size}
        ));
        const TensorPrecision c_attn_precision = require_linear_weight(
            checkpoint,
            prefix + ".attn.c_attn.weight",
            {embedding_size, qkv_size},
            1
        );
        static_cast<void>(require_tensor(
            checkpoint,
            prefix + ".attn.c_attn.bias",
            {qkv_size}
        ));
        const TensorPrecision attn_proj_precision = require_linear_weight(
            checkpoint,
            prefix + ".attn.c_proj.weight",
            {embedding_size, embedding_size},
            1
        );
        static_cast<void>(require_tensor(
            checkpoint,
            prefix + ".attn.c_proj.bias",
            {embedding_size}
        ));
        static_cast<void>(require_tensor(
            checkpoint,
            prefix + ".ln_2.weight",
            {embedding_size}
        ));
        static_cast<void>(require_tensor(
            checkpoint,
            prefix + ".ln_2.bias",
            {embedding_size}
        ));
        const TensorPrecision mlp_fc_precision = require_linear_weight(
            checkpoint,
            prefix + ".mlp.c_fc.weight",
            {embedding_size, feed_forward_size},
            1
        );
        static_cast<void>(require_tensor(
            checkpoint,
            prefix + ".mlp.c_fc.bias",
            {feed_forward_size}
        ));
        const TensorPrecision mlp_proj_precision = require_linear_weight(
            checkpoint,
            prefix + ".mlp.c_proj.weight",
            {feed_forward_size, embedding_size},
            1
        );
        static_cast<void>(require_tensor(
            checkpoint,
            prefix + ".mlp.c_proj.bias",
            {embedding_size}
        ));

        if (c_attn_precision != attn_proj_precision ||
            c_attn_precision != mlp_fc_precision ||
            c_attn_precision != mlp_proj_precision) {
            throw std::invalid_argument(
                "checkpoint mixes float32 and int8 weights within layer " +
                std::to_string(layer)
            );
        }

        if (c_attn_precision == TensorPrecision::int8) {
            quantized_tensor_count += 4;
        }

        if (!transformer_weight_precision_known) {
            transformer_weight_precision = c_attn_precision;
            transformer_weight_precision_known = true;
        } else if (c_attn_precision != transformer_weight_precision) {
            throw std::invalid_argument(
                "checkpoint quantizes some transformer layers but not "
                "others"
            );
        }
    }

    static_cast<void>(require_tensor(
        checkpoint,
        "transformer.ln_f.weight",
        {embedding_size}
    ));
    static_cast<void>(require_tensor(
        checkpoint,
        "transformer.ln_f.bias",
        {embedding_size}
    ));

    const std::size_t expected_tensor_count =
        layer_count * tensors_per_layer + top_level_tensor_count +
        quantized_tensor_count;
    if (checkpoint.tensor_count() != expected_tensor_count) {
        throw std::invalid_argument(
            "checkpoint tensor count does not match the GPT-2 schema"
        );
    }
}

TransformerBlockParameters bind_block(
    const Checkpoint& checkpoint,
    std::size_t layer
) {
    const std::string prefix = layer_prefix(layer);

    return {
        {
            checkpoint.tensor(prefix + ".ln_1.weight"),
            checkpoint.tensor(prefix + ".ln_1.bias"),
        },
        {
            {
                checkpoint.tensor(prefix + ".attn.c_attn.weight"),
                checkpoint.tensor(prefix + ".attn.c_attn.bias"),
            },
            {
                checkpoint.tensor(prefix + ".attn.c_proj.weight"),
                checkpoint.tensor(prefix + ".attn.c_proj.bias"),
            },
        },
        {
            checkpoint.tensor(prefix + ".ln_2.weight"),
            checkpoint.tensor(prefix + ".ln_2.bias"),
        },
        {
            {
                checkpoint.tensor(prefix + ".mlp.c_fc.weight"),
                checkpoint.tensor(prefix + ".mlp.c_fc.bias"),
            },
            {
                checkpoint.tensor(prefix + ".mlp.c_proj.weight"),
                checkpoint.tensor(prefix + ".mlp.c_proj.bias"),
            },
        },
    };
}

// Like bind_block, but every linear weight is read as its int8 form
// plus its ".quant_scale" companion. Callers must already have
// confirmed (via validate_checkpoint_schema, at construction) that
// this layer's four linear weights are all int8 -- this does not
// re-check that, so calling it on an FP32 layer throws whatever
// Checkpoint::int8_tensor throws for a name it does not hold.
QuantizedTransformerBlockParameters bind_quantized_block(
    const Checkpoint& checkpoint,
    std::size_t layer
) {
    const std::string prefix = layer_prefix(layer);

    return {
        {
            checkpoint.tensor(prefix + ".ln_1.weight"),
            checkpoint.tensor(prefix + ".ln_1.bias"),
        },
        {
            {
                checkpoint.int8_tensor(prefix + ".attn.c_attn.weight"),
                checkpoint.tensor(
                    prefix + ".attn.c_attn.weight.quant_scale"
                ),
                checkpoint.tensor(prefix + ".attn.c_attn.bias"),
            },
            {
                checkpoint.int8_tensor(prefix + ".attn.c_proj.weight"),
                checkpoint.tensor(
                    prefix + ".attn.c_proj.weight.quant_scale"
                ),
                checkpoint.tensor(prefix + ".attn.c_proj.bias"),
            },
        },
        {
            checkpoint.tensor(prefix + ".ln_2.weight"),
            checkpoint.tensor(prefix + ".ln_2.bias"),
        },
        {
            {
                checkpoint.int8_tensor(prefix + ".mlp.c_fc.weight"),
                checkpoint.tensor(
                    prefix + ".mlp.c_fc.weight.quant_scale"
                ),
                checkpoint.tensor(prefix + ".mlp.c_fc.bias"),
            },
            {
                checkpoint.int8_tensor(prefix + ".mlp.c_proj.weight"),
                checkpoint.tensor(
                    prefix + ".mlp.c_proj.weight.quant_scale"
                ),
                checkpoint.tensor(prefix + ".mlp.c_proj.bias"),
            },
        },
    };
}

// Runs one transformer layer, dispatching to the FP32 or quantized
// block implementation according to whatever this layer's own
// tensors are; validate_checkpoint_schema has already confirmed all
// four of a layer's linear weights agree, so checking just one
// (c_attn) is enough to know which of the two this layer is.
Tensor run_transformer_layer(
    const Checkpoint& checkpoint,
    std::size_t layer,
    const Tensor& hidden_state,
    std::size_t head_count
) {
    const std::string prefix = layer_prefix(layer);

    if (checkpoint.contains_int8(prefix + ".attn.c_attn.weight")) {
        return quantized_transformer_block(
            hidden_state,
            bind_quantized_block(checkpoint, layer),
            head_count
        );
    }

    return transformer_block(
        hidden_state,
        bind_block(checkpoint, layer),
        head_count
    );
}

Tensor run_cached_transformer_layer(
    const Checkpoint& checkpoint,
    std::size_t layer,
    const Tensor& hidden_state,
    std::size_t head_count,
    AttentionCache& cache
) {
    const std::string prefix = layer_prefix(layer);

    if (checkpoint.contains_int8(prefix + ".attn.c_attn.weight")) {
        return quantized_transformer_block(
            hidden_state,
            bind_quantized_block(checkpoint, layer),
            head_count,
            cache
        );
    }

    return transformer_block(
        hidden_state,
        bind_block(checkpoint, layer),
        head_count,
        cache
    );
}

Tensor tied_embedding_logits(
    const Tensor& hidden_state,
    const Tensor& token_embeddings
) {
    if (hidden_state.rank() != 2 || token_embeddings.rank() != 2) {
        throw std::invalid_argument(
            "tied output projection requires rank-2 tensors"
        );
    }

    const std::size_t sequence_length = hidden_state.shape()[0];
    const std::size_t embedding_size = hidden_state.shape()[1];
    const std::size_t vocabulary_size = token_embeddings.shape()[0];

    if (token_embeddings.shape()[1] != embedding_size) {
        throw std::invalid_argument(
            "token embedding width must match the hidden-state width"
        );
    }

    Tensor logits({sequence_length, vocabulary_size});

    const float* hidden_data = hidden_state.data();
    const float* embedding_data = token_embeddings.data();
    float* logits_data = logits.data();

    for (std::size_t token = 0; token < sequence_length; ++token) {
        for (std::size_t vocabulary_index = 0;
             vocabulary_index < vocabulary_size;
             ++vocabulary_index) {
            float sum = 0.0F;

            for (std::size_t feature = 0;
                 feature < embedding_size;
                 ++feature) {
                sum +=
                    hidden_data[token * embedding_size + feature] *
                    embedding_data[
                        vocabulary_index * embedding_size + feature
                    ];
            }

            logits_data[token * vocabulary_size + vocabulary_index] = sum;
        }
    }

    return logits;
}

// Like tied_embedding_logits, but token_embeddings is read as its int8
// form: each element is dequantized (cast, then scaled by its row's
// own entry in token_embeddings_scale) as it is consumed, rather than
// materializing a dequantized copy of the whole table first -- the
// same inline-dequantize approach quantized_linear/quantized_matmul
// use. wte's channel axis is its rows (one scale per vocabulary word),
// so the scale here is indexed by vocabulary_index, unlike
// quantized_linear's per-output-column indexing; see
// docs/quantization.md.
Tensor quantized_tied_embedding_logits(
    const Tensor& hidden_state,
    const Int8Tensor& token_embeddings,
    const Tensor& token_embeddings_scale
) {
    if (hidden_state.rank() != 2 || token_embeddings.rank() != 2) {
        throw std::invalid_argument(
            "tied output projection requires rank-2 tensors"
        );
    }

    const std::size_t sequence_length = hidden_state.shape()[0];
    const std::size_t embedding_size = hidden_state.shape()[1];
    const std::size_t vocabulary_size = token_embeddings.shape()[0];

    if (token_embeddings.shape()[1] != embedding_size) {
        throw std::invalid_argument(
            "token embedding width must match the hidden-state width"
        );
    }

    if (token_embeddings_scale.rank() != 1 ||
        token_embeddings_scale.numel() != vocabulary_size) {
        throw std::invalid_argument(
            "token embedding scale must have one entry per vocabulary "
            "word"
        );
    }

    Tensor logits({sequence_length, vocabulary_size});

    const float* hidden_data = hidden_state.data();
    const std::int8_t* embedding_data = token_embeddings.data();
    const float* scale_data = token_embeddings_scale.data();
    float* logits_data = logits.data();

    for (std::size_t token = 0; token < sequence_length; ++token) {
        for (std::size_t vocabulary_index = 0;
             vocabulary_index < vocabulary_size;
             ++vocabulary_index) {
            const float scale = scale_data[vocabulary_index];
            float sum = 0.0F;

            for (std::size_t feature = 0;
                 feature < embedding_size;
                 ++feature) {
                sum +=
                    hidden_data[token * embedding_size + feature] *
                    static_cast<float>(
                        embedding_data[
                            vocabulary_index * embedding_size + feature
                        ]
                    ) * scale;
            }

            logits_data[token * vocabulary_size + vocabulary_index] = sum;
        }
    }

    return logits;
}

// Reads the token embedding table, dispatching to the int8 path when
// wte was quantized and the FP32 path otherwise.
Tensor lookup_token_embeddings(
    const Checkpoint& checkpoint,
    std::span<const std::size_t> token_ids
) {
    if (checkpoint.contains_int8("transformer.wte.weight")) {
        return quantized_embedding_lookup(
            checkpoint.int8_tensor("transformer.wte.weight"),
            checkpoint.tensor("transformer.wte.weight.quant_scale"),
            token_ids
        );
    }

    return embedding_lookup(
        checkpoint.tensor("transformer.wte.weight"),
        token_ids
    );
}

// Projects a hidden state onto the vocabulary through the tied
// embedding, dispatching to the int8 path when wte was quantized and
// the FP32 path otherwise. wte's own precision is independent of the
// transformer layers' -- config 2 (see docs/quantization.md) keeps wte
// FP32 while every layer is int8.
Tensor compute_tied_logits(
    const Checkpoint& checkpoint,
    const Tensor& hidden_state
) {
    if (checkpoint.contains_int8("transformer.wte.weight")) {
        return quantized_tied_embedding_logits(
            hidden_state,
            checkpoint.int8_tensor("transformer.wte.weight"),
            checkpoint.tensor("transformer.wte.weight.quant_scale")
        );
    }

    return tied_embedding_logits(
        hidden_state,
        checkpoint.tensor("transformer.wte.weight")
    );
}

// Copies out the final row of a [sequence length, embedding size]
// hidden state as its own [1, embedding size] tensor. Generation never
// reads any row of forward's logits but the last one, so tied_
// embedding_logits is given only that row rather than the whole
// sequence; see docs/profiling.md.
Tensor extract_last_row(const Tensor& hidden_state) {
    const std::size_t sequence_length = hidden_state.shape()[0];
    const std::size_t embedding_size = hidden_state.shape()[1];

    Tensor last_row({1, embedding_size});
    std::copy_n(
        hidden_state.data() + (sequence_length - 1) * embedding_size,
        embedding_size,
        last_row.data()
    );

    return last_row;
}

}  // namespace

Gpt2Model::Gpt2Model(Checkpoint checkpoint)
    : checkpoint_m(std::move(checkpoint)),
      identity_m(next_model_identity()) {
    validate_checkpoint_schema(checkpoint_m);
}

const ModelConfig& Gpt2Model::config() const {
    return checkpoint_m.config();
}

namespace {

void validate_token_sequence(
    std::span<const std::size_t> token_ids,
    std::size_t available
) {
    if (token_ids.empty()) {
        throw std::invalid_argument(
            "GPT-2 forward pass requires at least one token"
        );
    }

    if (token_ids.size() > available) {
        throw std::invalid_argument(
            "token sequence exceeds the checkpoint context length"
        );
    }
}

}  // namespace

KvCache::KvCache(const ModelConfig& config)
    : embedding_size_m(config.embedding_size),
      head_count_m(config.head_count) {
    layers_m.reserve(static_cast<std::size_t>(config.layer_count));
    for (std::uint32_t layer = 0; layer < config.layer_count; ++layer) {
        layers_m.emplace_back(
            static_cast<std::size_t>(config.context_length),
            static_cast<std::size_t>(config.embedding_size)
        );
    }
}

std::size_t KvCache::length() const {
    return layers_m.empty() ? 0 : layers_m.front().length();
}

std::size_t KvCache::capacity() const {
    return layers_m.empty() ? 0 : layers_m.front().capacity();
}

std::size_t KvCache::layer_count() const {
    return layers_m.size();
}

void KvCache::clear() {
    for (AttentionCache& layer : layers_m) {
        layer.clear();
    }
    owner_id_m = 0;
}

Tensor Gpt2Model::run_transformer_stack(
    std::span<const std::size_t> token_ids
) const {
    const std::size_t context_length =
        static_cast<std::size_t>(config().context_length);
    validate_token_sequence(token_ids, context_length);

    std::vector<std::size_t> position_ids(token_ids.size());
    std::iota(position_ids.begin(), position_ids.end(), std::size_t{0});

    const Tensor token_state =
        lookup_token_embeddings(checkpoint_m, token_ids);
    const Tensor position_state = embedding_lookup(
        checkpoint_m.tensor("transformer.wpe.weight"),
        position_ids
    );
    Tensor hidden_state = add(token_state, position_state);

    const std::size_t layer_count =
        static_cast<std::size_t>(config().layer_count);
    const std::size_t head_count =
        static_cast<std::size_t>(config().head_count);

    for (std::size_t layer = 0; layer < layer_count; ++layer) {
        hidden_state = run_transformer_layer(
            checkpoint_m,
            layer,
            hidden_state,
            head_count
        );
    }

    return layer_norm(
        hidden_state,
        checkpoint_m.tensor("transformer.ln_f.weight"),
        checkpoint_m.tensor("transformer.ln_f.bias")
    );
}

Tensor Gpt2Model::forward(
    std::span<const std::size_t> token_ids
) const {
    return compute_tied_logits(
        checkpoint_m,
        run_transformer_stack(token_ids)
    );
}

Tensor Gpt2Model::forward_last_token_logits(
    std::span<const std::size_t> token_ids
) const {
    return compute_tied_logits(
        checkpoint_m,
        extract_last_row(run_transformer_stack(token_ids))
    );
}

// Runs every transformer layer, mutating cache in place, and returns
// the final (post layer_norm) hidden state for token_ids. Does not
// Throws before touching cache.layers_m in any way, so a rejected call
// never needs a rollback — see run_cached_transformer_stack below.
std::size_t Gpt2Model::validate_cached_forward(
    std::span<const std::size_t> token_ids,
    const KvCache& cache
) const {
    const std::size_t layer_count =
        static_cast<std::size_t>(config().layer_count);
    const std::size_t context_length =
        static_cast<std::size_t>(config().context_length);

    if (cache.layer_count() != layer_count ||
        cache.capacity() != context_length ||
        cache.embedding_size_m != config().embedding_size ||
        cache.head_count_m != config().head_count) {
        throw std::invalid_argument(
            "cache was not built for this model configuration"
        );
    }

    if (cache.owner_id_m != 0 && cache.owner_id_m != identity_m) {
        throw std::invalid_argument(
            "cache contains state from a different model"
        );
    }

    const std::size_t start = cache.length();
    for (const AttentionCache& layer : cache.layers_m) {
        if (layer.length() != start) {
            throw std::logic_error(
                "cache layers contain different sequence lengths"
            );
        }
    }
    validate_token_sequence(token_ids, context_length - start);

    return start;
}

// Runs every transformer layer, mutating cache in place, and returns
// the final (post layer_norm) hidden state for token_ids. The caller
// must have already validated the call with validate_cached_forward
// and pass its result as sequence_start; if this throws partway
// through, cache.layers_m may hold a mix of updated and un-updated
// lengths, and the caller is expected to roll every layer back to
// sequence_start.
Tensor Gpt2Model::run_cached_transformer_stack(
    std::span<const std::size_t> token_ids,
    KvCache& cache,
    std::size_t sequence_start
) const {
    // The cached tokens already occupy positions 0 through
    // sequence_start - 1, so the new tokens continue from there.
    std::vector<std::size_t> position_ids(token_ids.size());
    std::iota(
        position_ids.begin(),
        position_ids.end(),
        sequence_start
    );

    const Tensor token_state =
        lookup_token_embeddings(checkpoint_m, token_ids);
    const Tensor position_state = embedding_lookup(
        checkpoint_m.tensor("transformer.wpe.weight"),
        position_ids
    );
    Tensor hidden_state = add(token_state, position_state);

    const std::size_t layer_count =
        static_cast<std::size_t>(config().layer_count);
    const std::size_t head_count =
        static_cast<std::size_t>(config().head_count);

    for (std::size_t layer = 0; layer < layer_count; ++layer) {
        hidden_state = run_cached_transformer_layer(
            checkpoint_m,
            layer,
            hidden_state,
            head_count,
            cache.layers_m[layer]
        );
    }

    return layer_norm(
        hidden_state,
        checkpoint_m.tensor("transformer.ln_f.weight"),
        checkpoint_m.tensor("transformer.ln_f.bias")
    );
}

Tensor Gpt2Model::forward(
    std::span<const std::size_t> token_ids,
    KvCache& cache
) const {
    const std::size_t start = validate_cached_forward(token_ids, cache);
    try {
        Tensor logits = compute_tied_logits(
            checkpoint_m,
            run_cached_transformer_stack(token_ids, cache, start)
        );
        cache.owner_id_m = identity_m;
        return logits;
    } catch (...) {
        for (AttentionCache& layer : cache.layers_m) {
            layer.length_m = start;
        }
        throw;
    }
}

Tensor Gpt2Model::forward_last_token_logits(
    std::span<const std::size_t> token_ids,
    KvCache& cache
) const {
    const std::size_t start = validate_cached_forward(token_ids, cache);
    try {
        Tensor logits = compute_tied_logits(
            checkpoint_m,
            extract_last_row(
                run_cached_transformer_stack(token_ids, cache, start)
            )
        );
        cache.owner_id_m = identity_m;
        return logits;
    } catch (...) {
        for (AttentionCache& layer : cache.layers_m) {
            layer.length_m = start;
        }
        throw;
    }
}

}  // namespace gpt2
