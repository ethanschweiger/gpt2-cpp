#pragma once

#include "gpt2/tensor.h"

namespace gpt2 {

Tensor add(const Tensor& left, const Tensor& right);
Tensor matmul(const Tensor& left, const Tensor& right);

// Computes matmul(left, dequantize(right, right_scale)) without ever
// materializing that dequantized copy: each element of `right` is
// converted from int8 to float only as the inner loop reaches it. See
// docs/quantization.md for the scheme and why this project measures
// rather than assumes what that costs.
//
// `right_scale` holds one value per output column of `right` (its
// per-channel scale in the sense described there), so its length must
// equal right.shape()[1].
Tensor quantized_matmul(
    const Tensor& left,
    const Int8Tensor& right,
    const Tensor& right_scale
);

Tensor transpose(const Tensor& input);
Tensor softmax(const Tensor& input);

}  // namespace gpt2
