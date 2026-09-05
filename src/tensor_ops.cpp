#include "gpt2/tensor_ops.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
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

Tensor quantized_matmul(
    const Tensor& left,
    const Int8Tensor& right,
    const Tensor& right_scale
) {
    if (left.rank() != 2 || right.rank() != 2) {
        throw std::invalid_argument(
            "quantized matrix multiplication requires rank-2 tensors"
        );
    }

    const std::size_t rows = left.shape()[0];
    const std::size_t shared_size = left.shape()[1];
    const std::size_t right_rows = right.shape()[0];
    const std::size_t columns = right.shape()[1];

    if (shared_size != right_rows) {
        throw std::invalid_argument(
            "quantized matrix multiplication dimensions are incompatible"
        );
    }

    if (right_scale.rank() != 1 || right_scale.numel() != columns) {
        throw std::invalid_argument(
            "quantized matrix multiplication scale must have one "
            "value per output column"
        );
    }

    Tensor result({rows, columns});

    const float* left_data = left.data();
    const std::int8_t* right_data = right.data();
    const float* scale_data = right_scale.data();
    float* result_data = result.data();

    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            const float scale = scale_data[column];
            float sum = 0.0F;

            for (std::size_t inner = 0; inner < shared_size; ++inner) {
                const float dequantized_weight =
                    static_cast<float>(
                        right_data[inner * columns + column]
                    ) * scale;
                sum +=
                    left_data[row * shared_size + inner] *
                    dequantized_weight;
            }

            result_data[row * columns + column] = sum;
        }
    }

    return result;
}

Tensor transpose(const Tensor& input) {
    if (input.rank() != 2) {
        throw std::invalid_argument(
            "transpose requires a rank-2 tensor"
        );
    }

    const std::size_t rows = input.shape()[0];
    const std::size_t columns = input.shape()[1];

    Tensor result({columns, rows});

    const float* input_data = input.data();
    float* result_data = result.data();

    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            result_data[column * rows + row] =
                input_data[row * columns + column];
        }
    }

    return result;
}

Tensor softmax(const Tensor& input) {
    const std::size_t feature_count =
        input.shape().back();
    const std::size_t row_count =
        input.numel() / feature_count;

    Tensor result(input.shape());

    const float* input_data = input.data();
    float* result_data = result.data();

    for (std::size_t row = 0; row < row_count; ++row) {
        const std::size_t row_start =
            row * feature_count;

        float maximum = input_data[row_start];

        for (std::size_t feature = 1;
             feature < feature_count;
             ++feature) {
            const float value =
                input_data[row_start + feature];

            if (value > maximum) {
                maximum = value;
            }
        }

        float exponential_sum = 0.0F;

        for (std::size_t feature = 0;
             feature < feature_count;
             ++feature) {
            const std::size_t index =
                row_start + feature;

            result_data[index] =
                std::exp(input_data[index] - maximum);

            exponential_sum += result_data[index];
        }

        for (std::size_t feature = 0;
             feature < feature_count;
             ++feature) {
            result_data[row_start + feature] /=
                exponential_sum;
        }
    }

    return result;
}

}  // namespace gpt2
