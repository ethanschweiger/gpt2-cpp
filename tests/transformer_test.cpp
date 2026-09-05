#include "gpt2/transformer.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

int failure_count = 0;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failure_count;
    }
}

void expect_near(
    float actual,
    float expected,
    float tolerance,
    std::string_view message
) {
    if (!std::isfinite(actual) ||
        std::fabs(actual - expected) > tolerance) {
        std::cerr << "FAIL: " << message
                  << " (expected " << expected
                  << ", got " << actual << ")\n";
        ++failure_count;
    }
}

template <typename ExpectedException, typename Function>
void expect_throws(Function function, std::string_view message) {
    try {
        function();
    } catch (const ExpectedException&) {
        return;
    } catch (const std::exception& exception) {
        std::cerr << "FAIL: " << message
                  << " (wrong exception: " << exception.what() << ")\n";
        ++failure_count;
        return;
    } catch (...) {
        std::cerr << "FAIL: " << message
                  << " (wrong non-standard exception)\n";
        ++failure_count;
        return;
    }

    std::cerr << "FAIL: " << message << " (no exception thrown)\n";
    ++failure_count;
}

void test_feed_forward() {
    const gpt2::Tensor input(
        {1, 2},
        std::vector<float>{1.0F, -1.0F}
    );
    const gpt2::Tensor expansion_weight(
        {2, 3},
        std::vector<float>{
            1.0F, 0.0F, 1.0F,
            0.0F, 1.0F, 1.0F,
        }
    );
    const gpt2::Tensor expansion_bias({3});
    const gpt2::Tensor projection_weight(
        {3, 2},
        std::vector<float>{
            1.0F, 0.0F,
            0.0F, 1.0F,
            1.0F, 1.0F,
        }
    );
    const gpt2::Tensor projection_bias(
        {2},
        std::vector<float>{0.5F, -0.5F}
    );
    const gpt2::FeedForwardParameters parameters{
        {expansion_weight, expansion_bias},
        {projection_weight, projection_bias},
    };

    const gpt2::Tensor result =
        gpt2::feed_forward(input, parameters);

    expect(
        result.shape() == gpt2::Tensor::Shape{1, 2},
        "feed-forward network reduces back to the output width"
    );
    expect_near(
        result.at(0),
        1.341192F,
        1.0e-5F,
        "feed-forward network computes the first output"
    );
    expect_near(
        result.at(1),
        -0.658808F,
        1.0e-5F,
        "feed-forward network computes the second output"
    );
    expect(input.at(0) == 1.0F, "feed-forward leaves its input unchanged");
}

void test_zero_sublayers_preserve_residual() {
    const gpt2::Tensor input(
        {2, 2},
        std::vector<float>{1.0F, -1.0F, 2.0F, 4.0F}
    );
    const gpt2::Tensor norm_weight(
        {2},
        std::vector<float>{1.0F, 1.0F}
    );
    const gpt2::Tensor norm_bias({2});
    const gpt2::Tensor qkv_weight({2, 6});
    const gpt2::Tensor qkv_bias({6});
    const gpt2::Tensor attention_output_weight({2, 2});
    const gpt2::Tensor attention_output_bias({2});
    const gpt2::Tensor expansion_weight({2, 4});
    const gpt2::Tensor expansion_bias({4});
    const gpt2::Tensor projection_weight({4, 2});
    const gpt2::Tensor projection_bias({2});
    const gpt2::TransformerBlockParameters parameters{
        {norm_weight, norm_bias},
        {
            {qkv_weight, qkv_bias},
            {attention_output_weight, attention_output_bias},
        },
        {norm_weight, norm_bias},
        {
            {expansion_weight, expansion_bias},
            {projection_weight, projection_bias},
        },
    };

    const gpt2::Tensor result =
        gpt2::transformer_block(input, parameters, 2);

    expect(
        result.shape() == input.shape(),
        "transformer block preserves the residual shape"
    );

    for (std::size_t index = 0; index < input.numel(); ++index) {
        expect(
            result.at(index) == input.at(index),
            "zero sublayers leave the residual unchanged"
        );
    }
}

void test_transformer_block_matches_hugging_face_reference() {
    const gpt2::Tensor input(
        {3, 4},
        std::vector<float>{
            0.2F, -0.4F, 0.6F, 1.0F,
            1.2F, 0.3F, -0.7F, 0.5F,
            -0.5F, 0.8F, 0.1F, -1.1F,
        }
    );
    const gpt2::Tensor first_norm_weight(
        {4},
        std::vector<float>{1.1F, 0.9F, -1.2F, 0.8F}
    );
    const gpt2::Tensor first_norm_bias(
        {4},
        std::vector<float>{0.05F, -0.1F, 0.2F, 0.15F}
    );
    const gpt2::Tensor qkv_weight(
        {4, 12},
        std::vector<float>{
            0.4F, 0.1F, -0.2F, 0.0F,
            0.2F, -0.1F, 0.0F, 0.1F,
            0.5F, 0.1F, -0.2F, 0.3F,

            -0.1F, -0.3F, 0.0F, 0.2F,
            0.0F, 0.3F, 0.2F, -0.2F,
            -0.1F, 0.4F, 0.2F, -0.4F,

            0.2F, 0.0F, 0.3F, -0.1F,
            -0.3F, 0.1F, 0.4F, 0.0F,
            0.3F, -0.2F, 0.5F, 0.1F,

            0.0F, 0.2F, 0.1F, 0.5F,
            0.1F, 0.0F, -0.2F, 0.3F,
            0.2F, 0.3F, -0.1F, 0.6F,
        }
    );
    const gpt2::Tensor qkv_bias(
        {12},
        std::vector<float>{
            0.01F, -0.02F, 0.03F, -0.04F,
            0.05F, 0.06F, -0.07F, 0.08F,
            0.1F, -0.1F, 0.2F, -0.2F,
        }
    );
    const gpt2::Tensor attention_output_weight(
        {4, 4},
        std::vector<float>{
            0.6F, -0.1F, 0.2F, 0.0F,
            0.1F, 0.5F, -0.2F, 0.3F,
            -0.3F, 0.2F, 0.4F, 0.1F,
            0.2F, 0.0F, -0.1F, 0.7F,
        }
    );
    const gpt2::Tensor attention_output_bias(
        {4},
        std::vector<float>{0.03F, -0.04F, 0.05F, -0.06F}
    );
    const gpt2::Tensor second_norm_weight(
        {4},
        std::vector<float>{0.7F, -1.1F, 0.9F, 1.2F}
    );
    const gpt2::Tensor second_norm_bias(
        {4},
        std::vector<float>{-0.05F, 0.1F, 0.0F, 0.08F}
    );
    const gpt2::Tensor expansion_weight(
        {4, 8},
        std::vector<float>{
            0.2F, -0.1F, 0.3F, 0.0F,
            -0.2F, 0.4F, 0.1F, -0.3F,
            -0.4F, 0.2F, 0.1F, 0.3F,
            0.5F, -0.1F, 0.2F, 0.0F,
            0.1F, 0.3F, -0.2F, 0.4F,
            0.0F, 0.2F, -0.5F, 0.1F,
            0.3F, 0.0F, 0.2F, -0.1F,
            0.4F, -0.3F, 0.0F, 0.5F,
        }
    );
    const gpt2::Tensor expansion_bias(
        {8},
        std::vector<float>{
            0.01F, -0.02F, 0.03F, -0.04F,
            0.05F, -0.06F, 0.07F, -0.08F,
        }
    );
    const gpt2::Tensor projection_weight(
        {8, 4},
        std::vector<float>{
            0.2F, -0.1F, 0.0F, 0.3F,
            -0.3F, 0.4F, 0.1F, 0.0F,
            0.1F, 0.2F, -0.4F, 0.2F,
            0.0F, -0.2F, 0.3F, 0.1F,
            0.4F, 0.0F, 0.2F, -0.3F,
            -0.1F, 0.3F, 0.0F, 0.4F,
            0.2F, -0.4F, 0.5F, 0.0F,
            0.3F, 0.1F, -0.2F, 0.2F,
        }
    );
    const gpt2::Tensor projection_bias(
        {4},
        std::vector<float>{-0.02F, 0.03F, -0.01F, 0.04F}
    );
    const gpt2::TransformerBlockParameters parameters{
        {first_norm_weight, first_norm_bias},
        {
            {qkv_weight, qkv_bias},
            {attention_output_weight, attention_output_bias},
        },
        {second_norm_weight, second_norm_bias},
        {
            {expansion_weight, expansion_bias},
            {projection_weight, projection_bias},
        },
    };

    const gpt2::Tensor result =
        gpt2::transformer_block(input, parameters, 2);

    const std::array<float, 12> expected{
        1.392513990F, -0.608954608F, 0.763145745F, 1.196159244F,
        2.050967216F, -0.071960703F, -0.213598520F, 0.940857530F,
        -0.528725207F, 0.853673637F, 0.403173417F, -0.985633492F,
    };

    expect(
        result.shape() == gpt2::Tensor::Shape{3, 4},
        "transformer block preserves the reference input shape"
    );

    for (std::size_t index = 0; index < expected.size(); ++index) {
        expect_near(
            result.at(index),
            expected[index],
            1.0e-5F,
            "transformer block matches Hugging Face FP32 output"
        );
    }

    expect(input.at(0) == 0.2F, "transformer block leaves input unchanged");
    expect(
        qkv_weight.at(0) == 0.4F,
        "transformer block leaves parameters unchanged"
    );
}

void test_transformer_block_rejects_invalid_input() {
    const gpt2::Tensor input({2});
    const gpt2::Tensor vector({2});
    const gpt2::Tensor qkv_weight({2, 6});
    const gpt2::Tensor qkv_bias({6});
    const gpt2::Tensor matrix({2, 2});
    const gpt2::Tensor expansion_weight({2, 4});
    const gpt2::Tensor expansion_bias({4});
    const gpt2::Tensor projection_weight({4, 2});
    const gpt2::TransformerBlockParameters parameters{
        {vector, vector},
        {
            {qkv_weight, qkv_bias},
            {matrix, vector},
        },
        {vector, vector},
        {
            {expansion_weight, expansion_bias},
            {projection_weight, vector},
        },
    };

    expect_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                gpt2::transformer_block(input, parameters, 1)
            );
        },
        "transformer block rejects a non-matrix input"
    );
}

// A small deterministic sequence, distinct per call site (via `seed`),
// standing in for real trained weights: varied in sign and magnitude,
// reproducible, with no meaning beyond exercising every code path with
// numbers that are not all equal or all one sign.
std::int8_t deterministic_int8(std::size_t index, std::size_t seed) {
    const std::size_t encoded = (index * 37 + seed * 11) % 251;
    return static_cast<std::int8_t>(static_cast<int>(encoded) - 125);
}

std::vector<std::int8_t> make_deterministic_int8_values(
    std::size_t count,
    std::size_t seed
) {
    std::vector<std::int8_t> values(count);
    for (std::size_t index = 0; index < count; ++index) {
        values[index] = deterministic_int8(index, seed);
    }
    return values;
}

float deterministic_scale(std::size_t index, std::size_t seed) {
    return 0.01F + 0.003F *
        static_cast<float>((index * 7 + seed * 13) % 17);
}

std::vector<float> make_deterministic_scale_values(
    std::size_t count,
    std::size_t seed
) {
    std::vector<float> values(count);
    for (std::size_t index = 0; index < count; ++index) {
        values[index] = deterministic_scale(index, seed);
    }
    return values;
}

float deterministic_float(std::size_t index, std::size_t seed) {
    const std::size_t encoded = (index * 17 + seed * 5) % 23;
    return (static_cast<float>(encoded) - 11.0F) / 4.0F;
}

std::vector<float> make_deterministic_float_values(
    std::size_t count,
    std::size_t seed
) {
    std::vector<float> values(count);
    for (std::size_t index = 0; index < count; ++index) {
        values[index] = deterministic_float(index, seed);
    }
    return values;
}

gpt2::Tensor dequantize(
    const gpt2::Int8Tensor& quantized,
    const gpt2::Tensor& scale
) {
    const std::size_t rows = quantized.shape()[0];
    const std::size_t columns = quantized.shape()[1];
    gpt2::Tensor result({rows, columns});

    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            const std::array<std::size_t, 2> index{row, column};
            result.at(index) =
                static_cast<float>(
                    quantized.data()[row * columns + column]
                ) * scale.at(column);
        }
    }

    return result;
}

struct QuantizedTransformerBlockFixture {
    gpt2::Tensor input;
    gpt2::Tensor norm_weight;
    gpt2::Tensor norm_bias;
    gpt2::Int8Tensor qkv_weight;
    gpt2::Tensor qkv_scale;
    gpt2::Tensor qkv_bias;
    gpt2::Int8Tensor attention_output_weight;
    gpt2::Tensor attention_output_scale;
    gpt2::Tensor attention_output_bias;
    gpt2::Int8Tensor expansion_weight;
    gpt2::Tensor expansion_scale;
    gpt2::Tensor expansion_bias;
    gpt2::Int8Tensor projection_weight;
    gpt2::Tensor projection_scale;
    gpt2::Tensor projection_bias;
};

QuantizedTransformerBlockFixture make_quantized_transformer_block_fixture(
    std::size_t sequence_length,
    std::size_t embedding_size,
    std::size_t feed_forward_size
) {
    const std::size_t qkv_size = embedding_size * 3;

    return QuantizedTransformerBlockFixture{
        gpt2::Tensor(
            {sequence_length, embedding_size},
            make_deterministic_float_values(
                sequence_length * embedding_size, 0
            )
        ),
        gpt2::Tensor(
            {embedding_size},
            std::vector<float>(embedding_size, 1.0F)
        ),
        gpt2::Tensor({embedding_size}),
        gpt2::Int8Tensor(
            {embedding_size, qkv_size},
            make_deterministic_int8_values(embedding_size * qkv_size, 1)
        ),
        gpt2::Tensor({qkv_size}, make_deterministic_scale_values(qkv_size, 2)),
        gpt2::Tensor({qkv_size}, make_deterministic_float_values(qkv_size, 3)),
        gpt2::Int8Tensor(
            {embedding_size, embedding_size},
            make_deterministic_int8_values(embedding_size * embedding_size, 4)
        ),
        gpt2::Tensor(
            {embedding_size},
            make_deterministic_scale_values(embedding_size, 5)
        ),
        gpt2::Tensor(
            {embedding_size},
            make_deterministic_float_values(embedding_size, 6)
        ),
        gpt2::Int8Tensor(
            {embedding_size, feed_forward_size},
            make_deterministic_int8_values(
                embedding_size * feed_forward_size, 7
            )
        ),
        gpt2::Tensor(
            {feed_forward_size},
            make_deterministic_scale_values(feed_forward_size, 8)
        ),
        gpt2::Tensor(
            {feed_forward_size},
            make_deterministic_float_values(feed_forward_size, 9)
        ),
        gpt2::Int8Tensor(
            {feed_forward_size, embedding_size},
            make_deterministic_int8_values(
                feed_forward_size * embedding_size, 10
            )
        ),
        gpt2::Tensor(
            {embedding_size},
            make_deterministic_scale_values(embedding_size, 11)
        ),
        gpt2::Tensor(
            {embedding_size},
            make_deterministic_float_values(embedding_size, 12)
        ),
    };
}

gpt2::QuantizedTransformerBlockParameters as_quantized_parameters(
    const QuantizedTransformerBlockFixture& fixture
) {
    return gpt2::QuantizedTransformerBlockParameters{
        {fixture.norm_weight, fixture.norm_bias},
        {
            {fixture.qkv_weight, fixture.qkv_scale, fixture.qkv_bias},
            {
                fixture.attention_output_weight,
                fixture.attention_output_scale,
                fixture.attention_output_bias,
            },
        },
        {fixture.norm_weight, fixture.norm_bias},
        {
            {
                fixture.expansion_weight,
                fixture.expansion_scale,
                fixture.expansion_bias,
            },
            {
                fixture.projection_weight,
                fixture.projection_scale,
                fixture.projection_bias,
            },
        },
    };
}

gpt2::TransformerBlockParameters as_dequantized_parameters(
    const QuantizedTransformerBlockFixture& fixture,
    gpt2::Tensor& dequantized_qkv_weight,
    gpt2::Tensor& dequantized_attention_output_weight,
    gpt2::Tensor& dequantized_expansion_weight,
    gpt2::Tensor& dequantized_projection_weight
) {
    dequantized_qkv_weight =
        dequantize(fixture.qkv_weight, fixture.qkv_scale);
    dequantized_attention_output_weight = dequantize(
        fixture.attention_output_weight,
        fixture.attention_output_scale
    );
    dequantized_expansion_weight =
        dequantize(fixture.expansion_weight, fixture.expansion_scale);
    dequantized_projection_weight =
        dequantize(fixture.projection_weight, fixture.projection_scale);

    return gpt2::TransformerBlockParameters{
        {fixture.norm_weight, fixture.norm_bias},
        {
            {dequantized_qkv_weight, fixture.qkv_bias},
            {
                dequantized_attention_output_weight,
                fixture.attention_output_bias,
            },
        },
        {fixture.norm_weight, fixture.norm_bias},
        {
            {dequantized_expansion_weight, fixture.expansion_bias},
            {dequantized_projection_weight, fixture.projection_bias},
        },
    };
}

void test_quantized_feed_forward_matches_dequantized_reference() {
    const QuantizedTransformerBlockFixture fixture =
        make_quantized_transformer_block_fixture(2, 4, 6);
    const gpt2::QuantizedFeedForwardParameters quantized_parameters{
        {fixture.expansion_weight, fixture.expansion_scale, fixture.expansion_bias},
        {fixture.projection_weight, fixture.projection_scale, fixture.projection_bias},
    };
    const gpt2::Tensor dequantized_expansion_weight =
        dequantize(fixture.expansion_weight, fixture.expansion_scale);
    const gpt2::Tensor dequantized_projection_weight =
        dequantize(fixture.projection_weight, fixture.projection_scale);
    const gpt2::FeedForwardParameters plain_parameters{
        {dequantized_expansion_weight, fixture.expansion_bias},
        {dequantized_projection_weight, fixture.projection_bias},
    };

    const gpt2::Tensor quantized_result =
        gpt2::quantized_feed_forward(fixture.input, quantized_parameters);
    const gpt2::Tensor plain_result =
        gpt2::feed_forward(fixture.input, plain_parameters);

    for (std::size_t index = 0; index < quantized_result.numel(); ++index) {
        expect_near(
            quantized_result.at(index),
            plain_result.at(index),
            1.0e-3F,
            "quantized feed-forward matches feed-forward over "
            "explicitly dequantized weights"
        );
    }
}

void test_quantized_transformer_block_matches_dequantized_reference() {
    const QuantizedTransformerBlockFixture fixture =
        make_quantized_transformer_block_fixture(3, 4, 6);
    const gpt2::QuantizedTransformerBlockParameters quantized_parameters =
        as_quantized_parameters(fixture);

    gpt2::Tensor dequantized_qkv_weight({1});
    gpt2::Tensor dequantized_attention_output_weight({1});
    gpt2::Tensor dequantized_expansion_weight({1});
    gpt2::Tensor dequantized_projection_weight({1});
    const gpt2::TransformerBlockParameters plain_parameters =
        as_dequantized_parameters(
            fixture,
            dequantized_qkv_weight,
            dequantized_attention_output_weight,
            dequantized_expansion_weight,
            dequantized_projection_weight
        );

    const gpt2::Tensor quantized_result = gpt2::quantized_transformer_block(
        fixture.input,
        quantized_parameters,
        2
    );
    const gpt2::Tensor plain_result =
        gpt2::transformer_block(fixture.input, plain_parameters, 2);

    expect(
        quantized_result.shape() == plain_result.shape(),
        "quantized transformer block matches the plain result's shape"
    );
    for (std::size_t index = 0; index < quantized_result.numel(); ++index) {
        expect_near(
            quantized_result.at(index),
            plain_result.at(index),
            1.0e-2F,
            "quantized transformer block matches transformer_block "
            "over explicitly dequantized weights"
        );
    }
}

void test_quantized_transformer_block_cached_matches_uncached() {
    const QuantizedTransformerBlockFixture fixture =
        make_quantized_transformer_block_fixture(4, 4, 6);
    const gpt2::QuantizedTransformerBlockParameters parameters =
        as_quantized_parameters(fixture);
    const std::size_t sequence_length = fixture.input.shape()[0];
    const std::size_t embedding_size = fixture.input.shape()[1];

    const gpt2::Tensor uncached =
        gpt2::quantized_transformer_block(fixture.input, parameters, 2);

    gpt2::AttentionCache cache(sequence_length, embedding_size);
    gpt2::Tensor cached_result({sequence_length, embedding_size});
    for (std::size_t position = 0; position < sequence_length; ++position) {
        gpt2::Tensor step({1, embedding_size});
        for (std::size_t feature = 0; feature < embedding_size; ++feature) {
            const std::array<std::size_t, 2> step_index{0, feature};
            const std::array<std::size_t, 2> source_index{
                position, feature
            };
            step.at(step_index) = fixture.input.at(source_index);
        }

        const gpt2::Tensor step_result = gpt2::quantized_transformer_block(
            step,
            parameters,
            2,
            cache
        );

        for (std::size_t feature = 0; feature < embedding_size; ++feature) {
            const std::array<std::size_t, 2> destination_index{
                position, feature
            };
            cached_result.at(destination_index) =
                step_result.at(std::array<std::size_t, 2>{0, feature});
        }
    }

    for (std::size_t index = 0; index < uncached.numel(); ++index) {
        expect(
            uncached.at(index) == cached_result.at(index),
            "quantized transformer block's cached, one-token-at-a-time "
            "path reproduces the uncached result exactly"
        );
    }
}

void test_quantized_transformer_block_rejects_invalid_input() {
    const gpt2::Tensor input({2});
    const gpt2::Tensor vector({2});
    const gpt2::Int8Tensor qkv_weight(
        {2, 6},
        make_deterministic_int8_values(12, 20)
    );
    const gpt2::Tensor qkv_scale(
        {6},
        make_deterministic_scale_values(6, 21)
    );
    const gpt2::Int8Tensor matrix(
        {2, 2},
        make_deterministic_int8_values(4, 22)
    );
    const gpt2::Int8Tensor expansion_weight(
        {2, 4},
        make_deterministic_int8_values(8, 23)
    );
    const gpt2::Tensor expansion_scale(
        {4},
        make_deterministic_scale_values(4, 24)
    );
    const gpt2::Int8Tensor projection_weight(
        {4, 2},
        make_deterministic_int8_values(8, 25)
    );
    const gpt2::QuantizedTransformerBlockParameters parameters{
        {vector, vector},
        {
            {qkv_weight, qkv_scale, vector},
            {matrix, vector, vector},
        },
        {vector, vector},
        {
            {expansion_weight, expansion_scale, vector},
            {projection_weight, vector, vector},
        },
    };

    expect_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                gpt2::quantized_transformer_block(input, parameters, 1)
            );
        },
        "quantized transformer block rejects a non-matrix input"
    );
}

}  // namespace

int main() {
    test_feed_forward();
    test_zero_sublayers_preserve_residual();
    test_transformer_block_matches_hugging_face_reference();
    test_transformer_block_rejects_invalid_input();
    test_quantized_feed_forward_matches_dequantized_reference();
    test_quantized_transformer_block_matches_dequantized_reference();
    test_quantized_transformer_block_cached_matches_uncached();
    test_quantized_transformer_block_rejects_invalid_input();

    if (failure_count != 0) {
        std::cerr << failure_count << " transformer test assertion(s) failed\n";
        return EXIT_FAILURE;
    }

    std::cout << "All transformer tests passed\n";
    return EXIT_SUCCESS;
}
