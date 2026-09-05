#include "gpt2/layers.h"
#include "gpt2/tensor_ops.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace gpt2 {

Tensor embedding_lookup(
    const Tensor& embedding_table,
    std::span<const std::size_t> token_ids
) {
    if (embedding_table.rank() != 2) {
        throw std::invalid_argument(
            "embedding table must be a rank-2 tensor"
        );
    }

    if (token_ids.empty()) {
        throw std::invalid_argument(
            "embedding lookup requires at least one token"
        );
    }

    const std::size_t vocabulary_size =
        embedding_table.shape()[0];
    const std::size_t embedding_size =
        embedding_table.shape()[1];

    Tensor result({token_ids.size(), embedding_size});

    const float* table_data = embedding_table.data();
    float* result_data = result.data();

    for (std::size_t token_index = 0;
         token_index < token_ids.size();
         ++token_index) {
        const std::size_t token_id = token_ids[token_index];

        if (token_id >= vocabulary_size) {
            throw std::out_of_range(
                "token ID is outside the embedding vocabulary"
            );
        }

        const float* source =
            table_data + token_id * embedding_size;
        float* destination =
            result_data + token_index * embedding_size;

        std::copy_n(source, embedding_size, destination);
    }

    return result;
}

Tensor quantized_embedding_lookup(
    const Int8Tensor& embedding_table,
    const Tensor& embedding_scale,
    std::span<const std::size_t> token_ids
) {
    if (embedding_table.rank() != 2) {
        throw std::invalid_argument(
            "quantized embedding table must be a rank-2 tensor"
        );
    }

    const std::size_t vocabulary_size =
        embedding_table.shape()[0];
    const std::size_t embedding_size =
        embedding_table.shape()[1];

    if (embedding_scale.rank() != 1 ||
        embedding_scale.numel() != vocabulary_size) {
        throw std::invalid_argument(
            "quantized embedding scale must have one value per "
            "vocabulary entry"
        );
    }

    if (token_ids.empty()) {
        throw std::invalid_argument(
            "embedding lookup requires at least one token"
        );
    }

    Tensor result({token_ids.size(), embedding_size});

    const std::int8_t* table_data = embedding_table.data();
    const float* scale_data = embedding_scale.data();
    float* result_data = result.data();

    for (std::size_t token_index = 0;
         token_index < token_ids.size();
         ++token_index) {
        const std::size_t token_id = token_ids[token_index];

        if (token_id >= vocabulary_size) {
            throw std::out_of_range(
                "token ID is outside the embedding vocabulary"
            );
        }

        const std::int8_t* source =
            table_data + token_id * embedding_size;
        float* destination =
            result_data + token_index * embedding_size;
        const float scale = scale_data[token_id];

        for (std::size_t feature = 0;
             feature < embedding_size;
             ++feature) {
            destination[feature] =
                static_cast<float>(source[feature]) * scale;
        }
    }

    return result;
}

Tensor layer_norm(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    float epsilon
) {
    if (weight.rank() != 1 || bias.rank() != 1) {
        throw std::invalid_argument(
            "layer norm weight and bias must be rank-1 tensors"
        );
    }

    const std::size_t feature_count = input.shape().back();

    if (weight.numel() != feature_count ||
        bias.numel() != feature_count) {
        throw std::invalid_argument(
            "layer norm parameters must match the input's final dimension"
        );
    }

    if (!std::isfinite(epsilon) || epsilon <= 0.0F) {
        throw std::invalid_argument(
            "layer norm epsilon must be finite and positive"
        );
    }

    Tensor result(input.shape());

    const float* input_data = input.data();
    const float* weight_data = weight.data();
    const float* bias_data = bias.data();
    float* result_data = result.data();

    const std::size_t row_count =
        input.numel() / feature_count;

    for (std::size_t row = 0; row < row_count; ++row) {
        const std::size_t row_start = row * feature_count;

        float mean = 0.0F;

        for (std::size_t feature = 0;
             feature < feature_count;
             ++feature) {
            mean += input_data[row_start + feature];
        }

        mean /= static_cast<float>(feature_count);

        float variance = 0.0F;

        for (std::size_t feature = 0;
             feature < feature_count;
             ++feature) {
            const float difference =
                input_data[row_start + feature] - mean;

            variance += difference * difference;
        }

        variance /= static_cast<float>(feature_count);

        const float inverse_standard_deviation =
            1.0F / std::sqrt(variance + epsilon);

        for (std::size_t feature = 0;
             feature < feature_count;
             ++feature) {
            const std::size_t index = row_start + feature;
            const float normalized =
                (input_data[index] - mean) *
                inverse_standard_deviation;

            result_data[index] =
                normalized * weight_data[feature] +
                bias_data[feature];
        }
    }

    return result;
}

Tensor linear(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias
) {
    if (input.rank() != 2 || weight.rank() != 2) {
        throw std::invalid_argument(
            "linear input and weight must be rank-2 tensors"
        );
    }

    if (bias.rank() != 1) {
        throw std::invalid_argument(
            "linear bias must be a rank-1 tensor"
        );
    }

    const std::size_t input_feature_count =
        input.shape()[1];
    const std::size_t output_feature_count =
        weight.shape()[1];

    if (weight.shape()[0] != input_feature_count) {
        throw std::invalid_argument(
            "linear weight input size does not match the input"
        );
    }

    if (bias.numel() != output_feature_count) {
        throw std::invalid_argument(
            "linear bias size does not match the output size"
        );
    }

    Tensor result = matmul(input, weight);

    float* result_data = result.data();
    const float* bias_data = bias.data();
    const std::size_t row_count = input.shape()[0];

    for (std::size_t row = 0; row < row_count; ++row) {
        for (std::size_t output_feature = 0;
             output_feature < output_feature_count;
             ++output_feature) {
            const std::size_t index =
                row * output_feature_count + output_feature;

            result_data[index] += bias_data[output_feature];
        }
    }

    return result;
}

Tensor quantized_linear(
    const Tensor& input,
    const Int8Tensor& weight,
    const Tensor& weight_scale,
    const Tensor& bias
) {
    if (input.rank() != 2 || weight.rank() != 2) {
        throw std::invalid_argument(
            "quantized linear input and weight must be rank-2 tensors"
        );
    }

    if (bias.rank() != 1) {
        throw std::invalid_argument(
            "quantized linear bias must be a rank-1 tensor"
        );
    }

    const std::size_t input_feature_count =
        input.shape()[1];
    const std::size_t output_feature_count =
        weight.shape()[1];

    if (weight.shape()[0] != input_feature_count) {
        throw std::invalid_argument(
            "quantized linear weight input size does not match the "
            "input"
        );
    }

    if (bias.numel() != output_feature_count) {
        throw std::invalid_argument(
            "quantized linear bias size does not match the output "
            "size"
        );
    }

    // quantized_matmul checks weight_scale's own shape against
    // weight's, so this function need not repeat that check.
    Tensor result = quantized_matmul(input, weight, weight_scale);

    float* result_data = result.data();
    const float* bias_data = bias.data();
    const std::size_t row_count = input.shape()[0];

    for (std::size_t row = 0; row < row_count; ++row) {
        for (std::size_t output_feature = 0;
             output_feature < output_feature_count;
             ++output_feature) {
            const std::size_t index =
                row * output_feature_count + output_feature;

            result_data[index] += bias_data[output_feature];
        }
    }

    return result;
}

Tensor gelu(const Tensor& input) {
    constexpr float square_root_two_over_pi =
        0.7978845608028654F;
    constexpr float cubic_coefficient =
        0.044715F;

    Tensor result(input.shape());

    const float* input_data = input.data();
    float* result_data = result.data();

    for (std::size_t index = 0;
         index < input.numel();
         ++index) {
        const float value = input_data[index];
        const float value_cubed =
            value * value * value;

        result_data[index] =
            0.5F * value *
            (1.0F + std::tanh(
                square_root_two_over_pi *
                (value + cubic_coefficient * value_cubed)
            ));
    }

    return result;
}

}  // namespace gpt2
