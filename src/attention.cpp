#include "gpt2/attention.h"
#include "gpt2/layers.h"
#include "gpt2/tensor_ops.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace gpt2 {

namespace {

Tensor extract_head(
    const Tensor& combined_qkv,
    std::size_t component_offset,
    std::size_t head_index,
    std::size_t head_size
) {
    const std::size_t sequence_length =
        combined_qkv.shape()[0];
    const std::size_t combined_width =
        combined_qkv.shape()[1];

    Tensor result({sequence_length, head_size});

    const float* combined_data = combined_qkv.data();
    float* result_data = result.data();

    for (std::size_t token = 0;
         token < sequence_length;
         ++token) {
        const std::size_t source_start =
            token * combined_width +
            component_offset +
            head_index * head_size;
        const std::size_t destination_start =
            token * head_size;

        std::copy_n(
            combined_data + source_start,
            head_size,
            result_data + destination_start
        );
    }

    return result;
}

void store_head(
    Tensor& merged,
    const Tensor& head_output,
    std::size_t head_index
) {
    const std::size_t sequence_length =
        merged.shape()[0];
    const std::size_t embedding_size =
        merged.shape()[1];
    const std::size_t head_size =
        head_output.shape()[1];

    float* merged_data = merged.data();
    const float* head_data = head_output.data();

    for (std::size_t token = 0;
         token < sequence_length;
         ++token) {
        const std::size_t source_start =
            token * head_size;
        const std::size_t destination_start =
            token * embedding_size +
            head_index * head_size;

        std::copy_n(
            head_data + source_start,
            head_size,
            merged_data + destination_start
        );
    }
}

}  // namespace

Tensor causal_scaled_dot_product_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value
) {
    if (query.rank() != 2 ||
        key.rank() != 2 ||
        value.rank() != 2) {
        throw std::invalid_argument(
            "attention query, key, and value must be rank-2 tensors"
        );
    }

    if (query.shape() != key.shape() ||
        query.shape() != value.shape()) {
        throw std::invalid_argument(
            "attention query, key, and value shapes must match"
        );
    }

    const std::size_t sequence_length =
        query.shape()[0];
    const std::size_t head_size =
        query.shape()[1];

    const Tensor transposed_key = transpose(key);
    Tensor scores = matmul(query, transposed_key);

    const float scale =
        1.0F / std::sqrt(static_cast<float>(head_size));

    float* score_data = scores.data();

    for (std::size_t query_position = 0;
         query_position < sequence_length;
         ++query_position) {
        for (std::size_t key_position = 0;
             key_position < sequence_length;
             ++key_position) {
            const std::size_t index =
                query_position * sequence_length +
                key_position;

            if (key_position > query_position) {
                score_data[index] =
                    -std::numeric_limits<float>::infinity();
            } else {
                score_data[index] *= scale;
            }
        }
    }

    const Tensor weights = softmax(scores);

    return matmul(weights, value);
}

Tensor multi_head_self_attention(
    const Tensor& input,
    const Tensor& qkv_weight,
    const Tensor& qkv_bias,
    const Tensor& output_weight,
    const Tensor& output_bias,
    std::size_t head_count
) {
    if (input.rank() != 2) {
        throw std::invalid_argument(
            "multi-head attention input must be rank 2"
        );
    }

    if (head_count == 0) {
        throw std::invalid_argument(
            "multi-head attention requires at least one head"
        );
    }

    const std::size_t sequence_length =
        input.shape()[0];
    const std::size_t embedding_size =
        input.shape()[1];

    if (embedding_size % head_count != 0) {
        throw std::invalid_argument(
            "embedding size must be divisible by head count"
        );
    }

    if (embedding_size >
        std::numeric_limits<std::size_t>::max() / 3) {
        throw std::invalid_argument(
            "embedding size is too large for a QKV projection"
        );
    }

    const std::size_t qkv_size = embedding_size * 3;
    const std::size_t head_size =
        embedding_size / head_count;

    if (qkv_weight.rank() != 2 ||
        qkv_weight.shape()[0] != embedding_size ||
        qkv_weight.shape()[1] != qkv_size) {
        throw std::invalid_argument(
            "QKV weight must have shape [embedding size, 3 * embedding size]"
        );
    }

    if (qkv_bias.rank() != 1 ||
        qkv_bias.numel() != qkv_size) {
        throw std::invalid_argument(
            "QKV bias must have 3 * embedding size values"
        );
    }

    if (output_weight.rank() != 2 ||
        output_weight.shape()[0] != embedding_size ||
        output_weight.shape()[1] != embedding_size) {
        throw std::invalid_argument(
            "attention output weight must have shape [embedding size, embedding size]"
        );
    }

    if (output_bias.rank() != 1 ||
        output_bias.numel() != embedding_size) {
        throw std::invalid_argument(
            "attention output bias must match the embedding size"
        );
    }

    const Tensor combined_qkv =
        linear(input, qkv_weight, qkv_bias);

    Tensor merged({sequence_length, embedding_size});

    for (std::size_t head = 0;
         head < head_count;
         ++head) {
        const Tensor query = extract_head(
            combined_qkv,
            0,
            head,
            head_size
        );
        const Tensor key = extract_head(
            combined_qkv,
            embedding_size,
            head,
            head_size
        );
        const Tensor value = extract_head(
            combined_qkv,
            2 * embedding_size,
            head,
            head_size
        );

        const Tensor head_output =
            causal_scaled_dot_product_attention(
                query,
                key,
                value
            );

        store_head(merged, head_output, head);
    }

    return linear(
        merged,
        output_weight,
        output_bias
    );
}

}  // namespace gpt2
