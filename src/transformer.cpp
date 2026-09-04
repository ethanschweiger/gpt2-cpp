#include "gpt2/transformer.h"

#include "gpt2/attention.h"
#include "gpt2/layers.h"
#include "gpt2/tensor_ops.h"

#include <stdexcept>

namespace gpt2 {

Tensor feed_forward(
    const Tensor& input,
    const FeedForwardParameters& parameters
) {
    const Tensor expanded = linear(
        input,
        parameters.expansion.weight,
        parameters.expansion.bias
    );

    const Tensor activated = gelu(expanded);

    return linear(
        activated,
        parameters.projection.weight,
        parameters.projection.bias
    );
}

namespace {

Tensor run_block(
    const Tensor& input,
    const TransformerBlockParameters& parameters,
    const Tensor& attention_output
) {
    const Tensor after_attention = add(input, attention_output);

    const Tensor feed_forward_input = layer_norm(
        after_attention,
        parameters.feed_forward_norm.weight,
        parameters.feed_forward_norm.bias
    );

    const Tensor feed_forward_output = feed_forward(
        feed_forward_input,
        parameters.feed_forward
    );

    return add(after_attention, feed_forward_output);
}

}  // namespace

Tensor transformer_block(
    const Tensor& input,
    const TransformerBlockParameters& parameters,
    std::size_t head_count
) {
    if (input.rank() != 2) {
        throw std::invalid_argument(
            "transformer block input must be rank 2"
        );
    }

    const Tensor attention_input = layer_norm(
        input,
        parameters.attention_norm.weight,
        parameters.attention_norm.bias
    );

    const Tensor attention_output = multi_head_self_attention(
        attention_input,
        parameters.attention.qkv.weight,
        parameters.attention.qkv.bias,
        parameters.attention.output.weight,
        parameters.attention.output.bias,
        head_count
    );

    return run_block(input, parameters, attention_output);
}

Tensor transformer_block(
    const Tensor& input,
    const TransformerBlockParameters& parameters,
    std::size_t head_count,
    AttentionCache& cache
) {
    if (input.rank() != 2) {
        throw std::invalid_argument(
            "transformer block input must be rank 2"
        );
    }

    const Tensor attention_input = layer_norm(
        input,
        parameters.attention_norm.weight,
        parameters.attention_norm.bias
    );

    const Tensor attention_output = multi_head_self_attention(
        attention_input,
        parameters.attention.qkv.weight,
        parameters.attention.qkv.bias,
        parameters.attention.output.weight,
        parameters.attention.output.bias,
        head_count,
        cache
    );

    return run_block(input, parameters, attention_output);
}

}  // namespace gpt2
