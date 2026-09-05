#pragma once

#include "gpt2/tensor.h"

#include <cstddef>

namespace gpt2 {

class Gpt2Model;

Tensor causal_scaled_dot_product_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value
);

// The same attention with the queries starting partway through the
// sequence. Query row q stands at absolute position query_offset + q
// and may attend to every key at or before it, which is what lets a
// cached run score one new token against every earlier one.
Tensor causal_scaled_dot_product_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    std::size_t query_offset
);

// Keys and values already computed for the tokens seen so far. The
// tensors are sized for the whole context window and only the first
// `length` rows hold real values.
class AttentionCache {
public:
    AttentionCache(std::size_t capacity, std::size_t embedding_size);

    std::size_t length() const;
    std::size_t capacity() const;
    std::size_t embedding_size() const;
    void clear();

private:
    friend class Gpt2Model;
    friend Tensor multi_head_self_attention(
        const Tensor& input,
        const Tensor& qkv_weight,
        const Tensor& qkv_bias,
        const Tensor& output_weight,
        const Tensor& output_bias,
        std::size_t head_count,
        AttentionCache& cache
    );
    friend Tensor quantized_multi_head_self_attention(
        const Tensor& input,
        const Int8Tensor& qkv_weight,
        const Tensor& qkv_scale,
        const Tensor& qkv_bias,
        const Int8Tensor& output_weight,
        const Tensor& output_scale,
        const Tensor& output_bias,
        std::size_t head_count,
        AttentionCache& cache
    );

    Tensor keys_m;
    Tensor values_m;
    std::size_t length_m = 0;
};

Tensor multi_head_self_attention(
    const Tensor& input,
    const Tensor& qkv_weight,
    const Tensor& qkv_bias,
    const Tensor& output_weight,
    const Tensor& output_bias,
    std::size_t head_count
);

// Attention over the cached keys and values plus the new tokens in
// `input`. The new keys and values are appended to the cache, so the
// call both reads and extends it.
Tensor multi_head_self_attention(
    const Tensor& input,
    const Tensor& qkv_weight,
    const Tensor& qkv_bias,
    const Tensor& output_weight,
    const Tensor& output_bias,
    std::size_t head_count,
    AttentionCache& cache
);

// Like multi_head_self_attention, but the QKV and output projection
// weights are per-output-channel-quantized int8 tensors, dequantized
// as quantized_linear (layers.h) uses them rather than up front. Only
// the two linear projections differ from the FP32 path -- attention
// itself always runs on their FP32 output. See docs/quantization.md.
Tensor quantized_multi_head_self_attention(
    const Tensor& input,
    const Int8Tensor& qkv_weight,
    const Tensor& qkv_scale,
    const Tensor& qkv_bias,
    const Int8Tensor& output_weight,
    const Tensor& output_scale,
    const Tensor& output_bias,
    std::size_t head_count
);

Tensor quantized_multi_head_self_attention(
    const Tensor& input,
    const Int8Tensor& qkv_weight,
    const Tensor& qkv_scale,
    const Tensor& qkv_bias,
    const Int8Tensor& output_weight,
    const Tensor& output_scale,
    const Tensor& output_bias,
    std::size_t head_count,
    AttentionCache& cache
);

}  // namespace gpt2
