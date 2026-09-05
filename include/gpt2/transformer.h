#pragma once

#include "gpt2/attention.h"
#include "gpt2/tensor.h"

#include <cstddef>

namespace gpt2 {

// These parameter groups do not own or copy their tensors.
// Every referenced tensor must outlive calls that use the group.
struct LinearParameters {
    const Tensor& weight;
    const Tensor& bias;
};

struct LayerNormParameters {
    const Tensor& weight;
    const Tensor& bias;
};

struct AttentionParameters {
    LinearParameters qkv;
    LinearParameters output;
};

struct FeedForwardParameters {
    LinearParameters expansion;
    LinearParameters projection;
};

struct TransformerBlockParameters {
    LayerNormParameters attention_norm;
    AttentionParameters attention;
    LayerNormParameters feed_forward_norm;
    FeedForwardParameters feed_forward;
};

// Quantized siblings of the four structs above. A LayerNorm's weight
// and bias are always FP32 in this project (see docs/quantization.md),
// so LayerNormParameters is reused unchanged; only the four linear
// projections inside a block ever become QuantizedLinearParameters.
struct QuantizedLinearParameters {
    const Int8Tensor& weight;
    const Tensor& weight_scale;
    const Tensor& bias;
};

struct QuantizedAttentionParameters {
    QuantizedLinearParameters qkv;
    QuantizedLinearParameters output;
};

struct QuantizedFeedForwardParameters {
    QuantizedLinearParameters expansion;
    QuantizedLinearParameters projection;
};

struct QuantizedTransformerBlockParameters {
    LayerNormParameters attention_norm;
    QuantizedAttentionParameters attention;
    LayerNormParameters feed_forward_norm;
    QuantizedFeedForwardParameters feed_forward;
};

Tensor feed_forward(
    const Tensor& input,
    const FeedForwardParameters& parameters
);

Tensor quantized_feed_forward(
    const Tensor& input,
    const QuantizedFeedForwardParameters& parameters
);

Tensor transformer_block(
    const Tensor& input,
    const TransformerBlockParameters& parameters,
    std::size_t head_count
);

// The same block with attention reading and extending a cache, so the
// input holds only the tokens that have not been seen yet.
Tensor transformer_block(
    const Tensor& input,
    const TransformerBlockParameters& parameters,
    std::size_t head_count,
    AttentionCache& cache
);

// Like transformer_block(), but every linear projection in the block
// is per-output-channel-quantized int8, dequantized as
// quantized_linear (layers.h) uses it. See docs/quantization.md.
Tensor quantized_transformer_block(
    const Tensor& input,
    const QuantizedTransformerBlockParameters& parameters,
    std::size_t head_count
);

Tensor quantized_transformer_block(
    const Tensor& input,
    const QuantizedTransformerBlockParameters& parameters,
    std::size_t head_count,
    AttentionCache& cache
);

}  // namespace gpt2
