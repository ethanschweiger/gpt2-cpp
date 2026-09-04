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

// The cache stores whole rows for the context window, so a head is
// the same strided slice as in a freshly computed QKV block, limited to
// the rows that hold real values.
Tensor extract_cached_head(
    const Tensor& cached,
    std::size_t head_index,
    std::size_t head_size,
    std::size_t length
) {
    const std::size_t width = cached.shape()[1];

    Tensor result({length, head_size});

    const float* cached_data = cached.data();
    float* result_data = result.data();

    for (std::size_t token = 0; token < length; ++token) {
        std::copy_n(
            cached_data + token * width + head_index * head_size,
            head_size,
            result_data + token * head_size
        );
    }

    return result;
}

void append_to_cache(
    Tensor& cached,
    const Tensor& combined_qkv,
    std::size_t component_offset,
    std::size_t start_row
) {
    const std::size_t new_count = combined_qkv.shape()[0];
    const std::size_t combined_width = combined_qkv.shape()[1];
    const std::size_t width = cached.shape()[1];

    const float* source = combined_qkv.data();
    float* destination = cached.data();

    for (std::size_t token = 0; token < new_count; ++token) {
        std::copy_n(
            source + token * combined_width + component_offset,
            width,
            destination + (start_row + token) * width
        );
    }
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
    const Tensor& value,
    std::size_t query_offset
) {
    if (query.rank() != 2 ||
        key.rank() != 2 ||
        value.rank() != 2) {
        throw std::invalid_argument(
            "attention query, key, and value must be rank-2 tensors"
        );
    }

    if (key.shape() != value.shape()) {
        throw std::invalid_argument(
            "attention key and value shapes must match"
        );
    }

    if (query.shape()[1] != key.shape()[1]) {
        throw std::invalid_argument(
            "attention query and key head sizes must match"
        );
    }

    const std::size_t query_count = query.shape()[0];
    const std::size_t key_count = key.shape()[0];
    const std::size_t head_size = query.shape()[1];

    if (query_offset > key_count ||
        query_count > key_count - query_offset) {
        throw std::invalid_argument(
            "attention queries must sit inside the cached key range"
        );
    }

    const Tensor transposed_key = transpose(key);
    Tensor scores = matmul(query, transposed_key);

    const float scale =
        1.0F / std::sqrt(static_cast<float>(head_size));

    float* score_data = scores.data();

    for (std::size_t query_position = 0;
         query_position < query_count;
         ++query_position) {
        const std::size_t absolute_position =
            query_offset + query_position;

        for (std::size_t key_position = 0;
             key_position < key_count;
             ++key_position) {
            const std::size_t index =
                query_position * key_count + key_position;

            if (key_position > absolute_position) {
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

    return causal_scaled_dot_product_attention(query, key, value, 0);
}

AttentionCache::AttentionCache(
    std::size_t capacity,
    std::size_t embedding_size
)
    : keys_m({capacity, embedding_size}),
      values_m({capacity, embedding_size}) {
    if (capacity == 0 || embedding_size == 0) {
        throw std::invalid_argument(
            "an attention cache needs a positive capacity and width"
        );
    }
}

std::size_t AttentionCache::length() const {
    return length_m;
}

std::size_t AttentionCache::capacity() const {
    return keys_m.shape()[0];
}

std::size_t AttentionCache::embedding_size() const {
    return keys_m.shape()[1];
}

void AttentionCache::clear() {
    length_m = 0;
}

namespace {

struct AttentionShapes {
    std::size_t sequence_length;
    std::size_t embedding_size;
    std::size_t head_size;
};

AttentionShapes validate_attention(
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

    const std::size_t sequence_length = input.shape()[0];
    const std::size_t embedding_size = input.shape()[1];

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

    return AttentionShapes{
        sequence_length,
        embedding_size,
        embedding_size / head_count
    };
}

}  // namespace

Tensor multi_head_self_attention(
    const Tensor& input,
    const Tensor& qkv_weight,
    const Tensor& qkv_bias,
    const Tensor& output_weight,
    const Tensor& output_bias,
    std::size_t head_count
) {
    const AttentionShapes shapes = validate_attention(
        input,
        qkv_weight,
        qkv_bias,
        output_weight,
        output_bias,
        head_count
    );

    const Tensor combined_qkv =
        linear(input, qkv_weight, qkv_bias);

    Tensor merged({shapes.sequence_length, shapes.embedding_size});

    for (std::size_t head = 0; head < head_count; ++head) {
        const Tensor query = extract_head(
            combined_qkv,
            0,
            head,
            shapes.head_size
        );
        const Tensor key = extract_head(
            combined_qkv,
            shapes.embedding_size,
            head,
            shapes.head_size
        );
        const Tensor value = extract_head(
            combined_qkv,
            2 * shapes.embedding_size,
            head,
            shapes.head_size
        );

        const Tensor head_output =
            causal_scaled_dot_product_attention(query, key, value);

        store_head(merged, head_output, head);
    }

    return linear(merged, output_weight, output_bias);
}

Tensor multi_head_self_attention(
    const Tensor& input,
    const Tensor& qkv_weight,
    const Tensor& qkv_bias,
    const Tensor& output_weight,
    const Tensor& output_bias,
    std::size_t head_count,
    AttentionCache& cache
) {
    const AttentionShapes shapes = validate_attention(
        input,
        qkv_weight,
        qkv_bias,
        output_weight,
        output_bias,
        head_count
    );

    if (cache.embedding_size() != shapes.embedding_size) {
        throw std::invalid_argument(
            "attention cache width does not match the embedding size"
        );
    }

    if (cache.length_m > cache.capacity() ||
        shapes.sequence_length > cache.capacity() - cache.length_m) {
        throw std::invalid_argument(
            "attention cache has no room for the new tokens"
        );
    }

    const Tensor combined_qkv =
        linear(input, qkv_weight, qkv_bias);

    const std::size_t query_offset = cache.length_m;
    append_to_cache(
        cache.keys_m,
        combined_qkv,
        shapes.embedding_size,
        query_offset
    );
    append_to_cache(
        cache.values_m,
        combined_qkv,
        2 * shapes.embedding_size,
        query_offset
    );
    const std::size_t total = query_offset + shapes.sequence_length;

    Tensor merged({shapes.sequence_length, shapes.embedding_size});

    for (std::size_t head = 0; head < head_count; ++head) {
        const Tensor query = extract_head(
            combined_qkv,
            0,
            head,
            shapes.head_size
        );
        const Tensor key = extract_cached_head(
            cache.keys_m,
            head,
            shapes.head_size,
            total
        );
        const Tensor value = extract_cached_head(
            cache.values_m,
            head,
            shapes.head_size,
            total
        );

        const Tensor head_output =
            causal_scaled_dot_product_attention(
                query,
                key,
                value,
                query_offset
            );

        store_head(merged, head_output, head);
    }

    cache.length_m = total;
    return linear(merged, output_weight, output_bias);
}

}  // namespace gpt2
