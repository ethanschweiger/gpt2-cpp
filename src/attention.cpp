#include "gpt2/attention.h"
#include "gpt2/tensor_ops.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace gpt2 {

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

}  // namespace gpt2
