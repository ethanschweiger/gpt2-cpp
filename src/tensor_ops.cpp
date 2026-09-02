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

Tensor matmul(const Tensor& left, const Tensor& right) {
    if (left.rank() != 2 || right.rank() != 2) {
        throw std::invalid_argument(
            "matrix multiplication requires rank-2 tensors"
        );
    }

    const std::size_t rows = left.shape()[0];
    const std::size_t shared_size = left.shape()[1];
    const std::size_t right_rows = right.shape()[0];
    const std::size_t columns = right.shape()[1];

    if (shared_size != right_rows) {
        throw std::invalid_argument(
            "matrix multiplication dimensions are incompatible"
        );
    }

    Tensor result({rows, columns});

    const float* left_data = left.data();
    const float* right_data = right.data();
    float* result_data = result.data();

    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            float sum = 0.0F;

            for (std::size_t inner = 0; inner < shared_size; ++inner) {
                sum +=
                    left_data[row * shared_size + inner] *
                    right_data[inner * columns + column];
            }

            result_data[row * columns + column] = sum;
        }
    }

    return result;
}

}  // namespace gpt2
