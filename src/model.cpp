#include "gpt2/model.h"

#include "gpt2/layers.h"
#include "gpt2/tensor_ops.h"
#include "gpt2/transformer.h"

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
         top_level_tensor_count) / tensors_per_layer) {
        throw std::invalid_argument(
            "checkpoint layer count is too large"
        );
    }

    const std::size_t expected_tensor_count =
        layer_count * tensors_per_layer + top_level_tensor_count;
    if (checkpoint.tensor_count() != expected_tensor_count) {
        throw std::invalid_argument(
            "checkpoint tensor count does not match the GPT-2 schema"
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

    static_cast<void>(require_tensor(
        checkpoint,
        "transformer.wte.weight",
        {vocabulary_size, embedding_size}
    ));
    static_cast<void>(require_tensor(
        checkpoint,
        "transformer.wpe.weight",
        {context_length, embedding_size}
    ));

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
        static_cast<void>(require_tensor(
            checkpoint,
            prefix + ".attn.c_attn.weight",
            {embedding_size, qkv_size}
        ));
        static_cast<void>(require_tensor(
            checkpoint,
            prefix + ".attn.c_attn.bias",
            {qkv_size}
        ));
        static_cast<void>(require_tensor(
            checkpoint,
            prefix + ".attn.c_proj.weight",
            {embedding_size, embedding_size}
        ));
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
        static_cast<void>(require_tensor(
            checkpoint,
            prefix + ".mlp.c_fc.weight",
            {embedding_size, feed_forward_size}
        ));
        static_cast<void>(require_tensor(
            checkpoint,
            prefix + ".mlp.c_fc.bias",
            {feed_forward_size}
        ));
        static_cast<void>(require_tensor(
            checkpoint,
            prefix + ".mlp.c_proj.weight",
            {feed_forward_size, embedding_size}
        ));
        static_cast<void>(require_tensor(
            checkpoint,
            prefix + ".mlp.c_proj.bias",
            {embedding_size}
        ));
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

Tensor Gpt2Model::forward(
    std::span<const std::size_t> token_ids
) const {
    const std::size_t context_length =
        static_cast<std::size_t>(config().context_length);
    validate_token_sequence(token_ids, context_length);

    std::vector<std::size_t> position_ids(token_ids.size());
    std::iota(position_ids.begin(), position_ids.end(), std::size_t{0});

    const Tensor token_state = embedding_lookup(
        checkpoint_m.tensor("transformer.wte.weight"),
        token_ids
    );
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
        const TransformerBlockParameters parameters =
            bind_block(checkpoint_m, layer);

        hidden_state = transformer_block(
            hidden_state,
            parameters,
            head_count
        );
    }

    hidden_state = layer_norm(
        hidden_state,
        checkpoint_m.tensor("transformer.ln_f.weight"),
        checkpoint_m.tensor("transformer.ln_f.bias")
    );

    return tied_embedding_logits(
        hidden_state,
        checkpoint_m.tensor("transformer.wte.weight")
    );
}

Tensor Gpt2Model::forward(
    std::span<const std::size_t> token_ids,
    KvCache& cache
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

    // The cached tokens already occupy positions 0 through start - 1,
    // so the new tokens continue from there.
    std::vector<std::size_t> position_ids(token_ids.size());
    std::iota(position_ids.begin(), position_ids.end(), start);

    const Tensor token_state = embedding_lookup(
        checkpoint_m.tensor("transformer.wte.weight"),
        token_ids
    );
    const Tensor position_state = embedding_lookup(
        checkpoint_m.tensor("transformer.wpe.weight"),
        position_ids
    );
    Tensor hidden_state = add(token_state, position_state);

    const std::size_t head_count =
        static_cast<std::size_t>(config().head_count);

    try {
        for (std::size_t layer = 0; layer < layer_count; ++layer) {
            const TransformerBlockParameters parameters =
                bind_block(checkpoint_m, layer);

            hidden_state = transformer_block(
                hidden_state,
                parameters,
                head_count,
                cache.layers_m[layer]
            );
        }

        hidden_state = layer_norm(
            hidden_state,
            checkpoint_m.tensor("transformer.ln_f.weight"),
            checkpoint_m.tensor("transformer.ln_f.bias")
        );

        Tensor logits = tied_embedding_logits(
            hidden_state,
            checkpoint_m.tensor("transformer.wte.weight")
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
