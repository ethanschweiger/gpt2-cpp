#pragma once

#include "gpt2/tensor.h"

namespace gpt2 {

Tensor add(const Tensor& left, const Tensor& right);
Tensor matmul(const Tensor& left, const Tensor& right);
Tensor transpose(const Tensor& input);
Tensor softmax(const Tensor& input);

}  // namespace gpt2
