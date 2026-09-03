#pragma once

#include "gpt2/tensor.h"

#include <cstddef>
#include <span>

namespace gpt2 {

Tensor embedding_lookup(
    const Tensor& embedding_table,
    std::span<const std::size_t> token_ids
);

Tensor layer_norm(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    float epsilon = 1.0e-5F
);

Tensor gelu(const Tensor& input);

}  // namespace gpt2
