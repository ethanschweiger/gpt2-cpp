#include "gpt2/tensor_ops.h"

#include <stdexcept>

namespace gpt2 {

Tensor add(const Tensor& left, const Tensor& right) {
    if (left.shape() != right.shape()) {
        throw std::invalid_argument(
            "cannot add tensors with different shapes"
        );
    }

    Tensor result(left.shape());

    const float* left_data = left.data();
    const float* right_data = right.data();
    float* result_data = result.data();

    for (std::size_t index = 0; index < result.numel(); ++index) {
        result_data[index] = left_data[index] + right_data[index];
    }

    return result;
}

}  // namespace gpt2
