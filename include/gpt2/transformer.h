#pragma once

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

Tensor feed_forward(
    const Tensor& input,
    const FeedForwardParameters& parameters
);

Tensor transformer_block(
    const Tensor& input,
    const TransformerBlockParameters& parameters,
    std::size_t head_count
);

}  // namespace gpt2
