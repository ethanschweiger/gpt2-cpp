#pragma once

#include "gpt2/tensor.h"

#include <cstddef>
#include <span>

namespace gpt2 {

Tensor embedding_lookup(
    const Tensor& embedding_table,
    std::span<const std::size_t> token_ids
);

// Reads each looked-up row's int8 values through its own scale (one
// value per table row, so `embedding_scale` has one entry per
// vocabulary word) rather than a copy of the whole table dequantized
// up front. See docs/quantization.md.
Tensor quantized_embedding_lookup(
    const Int8Tensor& embedding_table,
    const Tensor& embedding_scale,
    std::span<const std::size_t> token_ids
);

Tensor layer_norm(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    float epsilon = 1.0e-5F
);

Tensor linear(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias
);

// Like linear(), but `weight` is a per-output-channel-quantized int8
// tensor: quantized_matmul (tensor_ops.h) dequantizes it as it is
// used rather than up front. `bias` is unquantized, as it always is
// in this project; see docs/quantization.md.
Tensor quantized_linear(
    const Tensor& input,
    const Int8Tensor& weight,
    const Tensor& weight_scale,
    const Tensor& bias
);

Tensor gelu(const Tensor& input);

}  // namespace gpt2
