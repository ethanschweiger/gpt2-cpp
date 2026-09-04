#pragma once

#include "gpt2/tensor.h"

#include <cstddef>

namespace gpt2 {

Tensor causal_scaled_dot_product_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value
);

Tensor multi_head_self_attention(
    const Tensor& input,
    const Tensor& qkv_weight,
    const Tensor& qkv_bias,
    const Tensor& output_weight,
    const Tensor& output_bias,
    std::size_t head_count
);

}  // namespace gpt2
