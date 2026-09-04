#pragma once

#include "gpt2/tensor.h"

namespace gpt2 {

Tensor causal_scaled_dot_product_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value
);

}  // namespace gpt2
