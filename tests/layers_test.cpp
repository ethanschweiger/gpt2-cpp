#include "gpt2/layers.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
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

void test_embedding_lookup() {
    const gpt2::Tensor table(
        {4, 3},
        std::vector<float>{
            0.1F, 0.2F, 0.3F,
            1.1F, 1.2F, 1.3F,
            2.1F, 2.2F, 2.3F,
            3.1F, 3.2F, 3.3F,
        }
    );
    const std::array<std::size_t, 4> token_ids{2, 0, 2, 3};

    const gpt2::Tensor result =
        gpt2::embedding_lookup(table, token_ids);

    expect(
        result.shape() == gpt2::Tensor::Shape{4, 3},
        "embedding lookup produces one row per token"
    );

    const std::array<float, 12> expected{
        2.1F, 2.2F, 2.3F,
        0.1F, 0.2F, 0.3F,
        2.1F, 2.2F, 2.3F,
        3.1F, 3.2F, 3.3F,
    };

    for (std::size_t index = 0; index < expected.size(); ++index) {
        expect(
            result.at(index) == expected[index],
            "embedding lookup copies the expected value"
        );
    }

    expect(table.at(0) == 0.1F, "embedding table is unchanged");
}

void test_invalid_embedding_table_rank() {
    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Tensor table({4});
            const std::array<std::size_t, 1> token_ids{0};
            static_cast<void>(
                gpt2::embedding_lookup(table, token_ids)
            );
        },
        "embedding lookup rejects a non-matrix table"
    );
}

void test_empty_token_sequence() {
    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Tensor table({2, 3});
            const std::vector<std::size_t> token_ids;
            static_cast<void>(
                gpt2::embedding_lookup(table, token_ids)
            );
        },
        "embedding lookup rejects an empty token sequence"
    );
}

void test_out_of_range_token_id() {
    expect_throws<std::out_of_range>(
        [] {
            const gpt2::Tensor table({2, 3});
            const std::array<std::size_t, 1> token_ids{2};
            static_cast<void>(
                gpt2::embedding_lookup(table, token_ids)
            );
        },
        "embedding lookup rejects an out-of-range token ID"
    );
}

void test_quantized_embedding_lookup() {
    // Every row's scale is 0.1, so int8 values 1..33 reproduce
    // test_embedding_lookup's exact FP32 table above with no rounding.
    const gpt2::Int8Tensor table(
        {4, 3},
        std::vector<std::int8_t>{
            1, 2, 3,
            11, 12, 13,
            21, 22, 23,
            31, 32, 33,
        }
    );
    const gpt2::Tensor scale(
        {4},
        std::vector<float>{0.1F, 0.1F, 0.1F, 0.1F}
    );
    const std::array<std::size_t, 4> token_ids{2, 0, 2, 3};

    const gpt2::Tensor result =
        gpt2::quantized_embedding_lookup(table, scale, token_ids);

    expect(
        result.shape() == gpt2::Tensor::Shape{4, 3},
        "quantized embedding lookup produces one row per token"
    );

    const std::array<float, 12> expected{
        2.1F, 2.2F, 2.3F,
        0.1F, 0.2F, 0.3F,
        2.1F, 2.2F, 2.3F,
        3.1F, 3.2F, 3.3F,
    };

    for (std::size_t index = 0; index < expected.size(); ++index) {
        expect_near(
            result.at(index),
            expected[index],
            1.0e-6F,
            "quantized embedding lookup dequantizes the expected value"
        );
    }

    expect(table.data()[0] == 1, "quantized embedding table is unchanged");
}

void test_quantized_embedding_lookup_uses_each_rows_own_scale() {
    // Every row holds the same int8 pattern; only a correct per-row
    // (not per-output-position) scale lookup can tell them apart.
    const gpt2::Int8Tensor table(
        {3, 2},
        std::vector<std::int8_t>{
            10, -10,
            10, -10,
            10, -10,
        }
    );
    const gpt2::Tensor scale(
        {3},
        std::vector<float>{0.1F, 1.0F, 10.0F}
    );
    // Deliberately out of row order, so a bug that indexed by output
    // position instead of token ID would read the wrong scale.
    const std::array<std::size_t, 3> token_ids{2, 0, 1};

    const gpt2::Tensor result =
        gpt2::quantized_embedding_lookup(table, scale, token_ids);

    const std::array<float, 6> expected{
        100.0F, -100.0F,
        1.0F, -1.0F,
        10.0F, -10.0F,
    };
    for (std::size_t index = 0; index < expected.size(); ++index) {
        expect_near(
            result.at(index),
            expected[index],
            1.0e-6F,
            "each looked-up row is dequantized with its own row's scale"
        );
    }
}

void test_invalid_quantized_embedding_table_rank() {
    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Int8Tensor table({4}, {1, 2, 3, 4});
            const gpt2::Tensor scale({4}, std::vector<float>(4, 1.0F));
            const std::array<std::size_t, 1> token_ids{0};
            static_cast<void>(
                gpt2::quantized_embedding_lookup(table, scale, token_ids)
            );
        },
        "quantized embedding lookup rejects a non-matrix table"
    );
}

void test_quantized_embedding_empty_token_sequence() {
    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Int8Tensor table({2, 3}, {1, 2, 3, 4, 5, 6});
            const gpt2::Tensor scale({2}, std::vector<float>{1.0F, 1.0F});
            const std::vector<std::size_t> token_ids;
            static_cast<void>(
                gpt2::quantized_embedding_lookup(table, scale, token_ids)
            );
        },
        "quantized embedding lookup rejects an empty token sequence"
    );
}

void test_quantized_embedding_out_of_range_token_id() {
    expect_throws<std::out_of_range>(
        [] {
            const gpt2::Int8Tensor table({2, 3}, {1, 2, 3, 4, 5, 6});
            const gpt2::Tensor scale({2}, std::vector<float>{1.0F, 1.0F});
            const std::array<std::size_t, 1> token_ids{2};
            static_cast<void>(
                gpt2::quantized_embedding_lookup(table, scale, token_ids)
            );
        },
        "quantized embedding lookup rejects an out-of-range token ID"
    );
}

void test_quantized_embedding_scale_shape_mismatch() {
    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Int8Tensor table({2, 3}, {1, 2, 3, 4, 5, 6});
            const gpt2::Tensor scale({3}, std::vector<float>(3, 1.0F));
            const std::array<std::size_t, 1> token_ids{0};
            static_cast<void>(
                gpt2::quantized_embedding_lookup(table, scale, token_ids)
            );
        },
        "quantized embedding lookup rejects a scale with the wrong length"
    );

    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Int8Tensor table({2, 3}, {1, 2, 3, 4, 5, 6});
            const gpt2::Tensor scale(
                {1, 2},
                std::vector<float>{1.0F, 1.0F}
            );
            const std::array<std::size_t, 1> token_ids{0};
            static_cast<void>(
                gpt2::quantized_embedding_lookup(table, scale, token_ids)
            );
        },
        "quantized embedding lookup rejects a scale that is not rank 1"
    );
}

void test_layer_norm_normalizes_rows_independently() {
    const gpt2::Tensor input(
        {2, 2},
        std::vector<float>{1.0F, 3.0F, 2.0F, 6.0F}
    );
    const gpt2::Tensor weight(
        {2},
        std::vector<float>{1.0F, 1.0F}
    );
    const gpt2::Tensor bias(
        {2},
        std::vector<float>{0.0F, 0.0F}
    );

    const gpt2::Tensor result =
        gpt2::layer_norm(input, weight, bias);

    expect(
        result.shape() == input.shape(),
        "layer norm preserves the input shape"
    );
    expect_near(
        result.at(0),
        -0.999995F,
        1.0e-5F,
        "first row's lower value is normalized"
    );
    expect_near(
        result.at(1),
        0.999995F,
        1.0e-5F,
        "first row's upper value is normalized"
    );
    expect_near(
        result.at(2),
        -0.999999F,
        1.0e-5F,
        "second row's lower value is normalized independently"
    );
    expect_near(
        result.at(3),
        0.999999F,
        1.0e-5F,
        "second row's upper value is normalized independently"
    );
    expect(input.at(0) == 1.0F, "layer norm leaves its input unchanged");
}

void test_layer_norm_applies_weight_and_bias() {
    const gpt2::Tensor input(
        {1, 2},
        std::vector<float>{1.0F, 3.0F}
    );
    const gpt2::Tensor weight(
        {2},
        std::vector<float>{2.0F, 0.5F}
    );
    const gpt2::Tensor bias(
        {2},
        std::vector<float>{10.0F, -3.0F}
    );

    const gpt2::Tensor result =
        gpt2::layer_norm(input, weight, bias);

    expect_near(
        result.at(0),
        8.00001F,
        1.0e-5F,
        "layer norm applies the first learned weight and bias"
    );
    expect_near(
        result.at(1),
        -2.5F,
        1.0e-5F,
        "layer norm applies the second learned weight and bias"
    );
}

void test_layer_norm_constant_row() {
    const gpt2::Tensor input(
        {1, 3},
        std::vector<float>{7.0F, 7.0F, 7.0F}
    );
    const gpt2::Tensor weight(
        {3},
        std::vector<float>{2.0F, 3.0F, 4.0F}
    );
    const gpt2::Tensor bias(
        {3},
        std::vector<float>{0.5F, -1.0F, 2.0F}
    );

    const gpt2::Tensor result =
        gpt2::layer_norm(input, weight, bias);

    expect(result.at(0) == 0.5F, "constant row returns first bias value");
    expect(result.at(1) == -1.0F, "constant row returns second bias value");
    expect(result.at(2) == 2.0F, "constant row returns third bias value");
}

void test_invalid_layer_norm_parameters() {
    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Tensor input({1, 2});
            const gpt2::Tensor weight({1, 2});
            const gpt2::Tensor bias({2});
            static_cast<void>(gpt2::layer_norm(input, weight, bias));
        },
        "layer norm rejects a non-vector weight"
    );

    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Tensor input({1, 2});
            const gpt2::Tensor weight({2});
            const gpt2::Tensor bias({1, 2});
            static_cast<void>(gpt2::layer_norm(input, weight, bias));
        },
        "layer norm rejects a non-vector bias"
    );

    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Tensor input({1, 2});
            const gpt2::Tensor weight({3});
            const gpt2::Tensor bias({2});
            static_cast<void>(gpt2::layer_norm(input, weight, bias));
        },
        "layer norm rejects a weight with the wrong length"
    );

    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Tensor input({1, 2});
            const gpt2::Tensor weight({2});
            const gpt2::Tensor bias({3});
            static_cast<void>(gpt2::layer_norm(input, weight, bias));
        },
        "layer norm rejects a bias with the wrong length"
    );
}

void test_invalid_layer_norm_epsilon() {
    const gpt2::Tensor input({1, 2});
    const gpt2::Tensor weight({2});
    const gpt2::Tensor bias({2});

    expect_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                gpt2::layer_norm(input, weight, bias, 0.0F)
            );
        },
        "layer norm rejects zero epsilon"
    );

    expect_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                gpt2::layer_norm(input, weight, bias, -1.0F)
            );
        },
        "layer norm rejects negative epsilon"
    );

    expect_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(gpt2::layer_norm(
                input,
                weight,
                bias,
                std::numeric_limits<float>::infinity()
            ));
        },
        "layer norm rejects infinite epsilon"
    );

    expect_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(gpt2::layer_norm(
                input,
                weight,
                bias,
                std::numeric_limits<float>::quiet_NaN()
            ));
        },
        "layer norm rejects NaN epsilon"
    );
}

void test_gelu_known_values() {
    const gpt2::Tensor input(
        {1, 5},
        std::vector<float>{-2.0F, -1.0F, 0.0F, 1.0F, 2.0F}
    );

    const gpt2::Tensor result = gpt2::gelu(input);

    const std::array<float, 5> expected{
        -0.0454023F,
        -0.158808F,
        0.0F,
        0.841192F,
        1.9546F,
    };

    expect(
        result.shape() == input.shape(),
        "GELU preserves the input shape"
    );

    for (std::size_t index = 0; index < expected.size(); ++index) {
        expect_near(
            result.at(index),
            expected[index],
            1.0e-5F,
            "GELU produces the expected value"
        );
    }

    expect(input.at(0) == -2.0F, "GELU leaves its input unchanged");
}

void test_gelu_multidimensional_tensor() {
    const gpt2::Tensor input(
        {2, 2, 2},
        std::vector<float>{
            -2.0F, -1.0F, 0.0F, 1.0F,
            2.0F, -0.5F, 0.5F, 3.0F,
        }
    );

    const gpt2::Tensor result = gpt2::gelu(input);

    expect(
        result.shape() == gpt2::Tensor::Shape{2, 2, 2},
        "GELU preserves a multidimensional shape"
    );
    expect_near(
        result.at(5),
        -0.154286F,
        1.0e-5F,
        "GELU processes a negative value in a multidimensional tensor"
    );
    expect_near(
        result.at(6),
        0.345714F,
        1.0e-5F,
        "GELU processes a positive value in a multidimensional tensor"
    );
    expect_near(
        result.at(7),
        2.99636F,
        1.0e-5F,
        "GELU processes every element in a multidimensional tensor"
    );
}

void test_linear_transformation() {
    const gpt2::Tensor input(
        {2, 3},
        std::vector<float>{
            1.0F, 2.0F, 3.0F,
            4.0F, 5.0F, 6.0F,
        }
    );
    const gpt2::Tensor weight(
        {3, 2},
        std::vector<float>{
            1.0F, 2.0F,
            3.0F, 4.0F,
            5.0F, 6.0F,
        }
    );
    const gpt2::Tensor bias(
        {2},
        std::vector<float>{0.5F, -1.0F}
    );

    const gpt2::Tensor result =
        gpt2::linear(input, weight, bias);

    expect(
        result.shape() == gpt2::Tensor::Shape{2, 2},
        "linear produces one output row per input row"
    );
    const std::array<float, 4> expected{22.5F, 27.0F, 49.5F, 63.0F};

    for (std::size_t index = 0; index < expected.size(); ++index) {
        expect_near(
            result.at(index),
            expected[index],
            1.0e-5F,
            "linear computes the expected output value"
        );
    }

    expect(input.at(0) == 1.0F, "linear leaves its input unchanged");
    expect(weight.at(0) == 1.0F, "linear leaves its weight unchanged");
    expect(bias.at(0) == 0.5F, "linear leaves its bias unchanged");
}

void test_linear_single_output_feature() {
    const gpt2::Tensor input(
        {2, 2},
        std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F}
    );
    const gpt2::Tensor weight(
        {2, 1},
        std::vector<float>{2.0F, -1.0F}
    );
    const gpt2::Tensor bias(
        {1},
        std::vector<float>{3.0F}
    );

    const gpt2::Tensor result =
        gpt2::linear(input, weight, bias);

    expect(
        result.shape() == gpt2::Tensor::Shape{2, 1},
        "linear supports a single output feature"
    );
    expect_near(
        result.at(0),
        3.0F,
        1.0e-5F,
        "linear adds bias to the first row"
    );
    expect_near(
        result.at(1),
        5.0F,
        1.0e-5F,
        "linear adds bias to the second row"
    );
}

void test_invalid_linear_tensor_ranks() {
    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Tensor input({3});
            const gpt2::Tensor weight({3, 2});
            const gpt2::Tensor bias({2});
            static_cast<void>(gpt2::linear(input, weight, bias));
        },
        "linear rejects a non-matrix input"
    );

    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Tensor input({1, 3});
            const gpt2::Tensor weight({3});
            const gpt2::Tensor bias({2});
            static_cast<void>(gpt2::linear(input, weight, bias));
        },
        "linear rejects a non-matrix weight"
    );

    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Tensor input({1, 3});
            const gpt2::Tensor weight({3, 2});
            const gpt2::Tensor bias({1, 2});
            static_cast<void>(gpt2::linear(input, weight, bias));
        },
        "linear rejects a non-vector bias"
    );
}

void test_invalid_linear_dimensions() {
    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Tensor input({2, 3});
            const gpt2::Tensor weight({4, 2});
            const gpt2::Tensor bias({2});
            static_cast<void>(gpt2::linear(input, weight, bias));
        },
        "linear rejects an incompatible weight input size"
    );

    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Tensor input({2, 3});
            const gpt2::Tensor weight({3, 2});
            const gpt2::Tensor bias({3});
            static_cast<void>(gpt2::linear(input, weight, bias));
        },
        "linear rejects an incompatible bias size"
    );
}

void test_quantized_linear_transformation() {
    const gpt2::Tensor input(
        {2, 3},
        std::vector<float>{
            1.0F, 2.0F, 3.0F,
            4.0F, 5.0F, 6.0F,
        }
    );
    // A scale of 1 everywhere makes the int8 weight exactly the same
    // matrix test_linear_transformation uses, so the two share their
    // expected output with no quantization error to account for.
    const gpt2::Int8Tensor weight(
        {3, 2},
        std::vector<std::int8_t>{
            1, 2,
            3, 4,
            5, 6,
        }
    );
    const gpt2::Tensor scale({2}, std::vector<float>{1.0F, 1.0F});
    const gpt2::Tensor bias(
        {2},
        std::vector<float>{0.5F, -1.0F}
    );

    const gpt2::Tensor result =
        gpt2::quantized_linear(input, weight, scale, bias);

    expect(
        result.shape() == gpt2::Tensor::Shape{2, 2},
        "quantized linear produces one output row per input row"
    );
    const std::array<float, 4> expected{22.5F, 27.0F, 49.5F, 63.0F};

    for (std::size_t index = 0; index < expected.size(); ++index) {
        expect_near(
            result.at(index),
            expected[index],
            1.0e-5F,
            "quantized linear computes the expected output value"
        );
    }

    expect(input.at(0) == 1.0F, "quantized linear leaves its input unchanged");
    expect(
        weight.data()[0] == 1,
        "quantized linear leaves its weight unchanged"
    );
    expect(bias.at(0) == 0.5F, "quantized linear leaves its bias unchanged");
}

void test_quantized_linear_matches_dequantize_then_linear() {
    const gpt2::Tensor input(
        {2, 3},
        std::vector<float>{
            -1.5F, 2.0F, 0.5F,
            3.0F, -4.0F, 1.0F,
        }
    );
    const std::vector<std::int8_t> quantized_values{
        100, -50,
        -20, 127,
        60, -110,
    };
    const gpt2::Int8Tensor weight({3, 2}, quantized_values);
    const gpt2::Tensor scale({2}, std::vector<float>{0.05F, 0.2F});
    const gpt2::Tensor bias({2}, std::vector<float>{1.5F, -2.5F});

    gpt2::Tensor dequantized_weight({3, 2});
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 2; ++column) {
            const std::array<std::size_t, 2> index{row, column};
            dequantized_weight.at(index) =
                static_cast<float>(quantized_values[row * 2 + column]) *
                scale.at(column);
        }
    }

    const gpt2::Tensor quantized_result =
        gpt2::quantized_linear(input, weight, scale, bias);
    const gpt2::Tensor plain_result =
        gpt2::linear(input, dequantized_weight, bias);

    for (std::size_t index = 0; index < quantized_result.numel(); ++index) {
        expect_near(
            quantized_result.at(index),
            plain_result.at(index),
            1.0e-4F,
            "quantized_linear matches linear over an explicitly "
            "dequantized copy of the same weight"
        );
    }
}

void test_invalid_quantized_linear_tensor_ranks() {
    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Tensor input({3});
            const gpt2::Int8Tensor weight({3, 2}, {1, 2, 3, 4, 5, 6});
            const gpt2::Tensor scale({2}, std::vector<float>{1.0F, 1.0F});
            const gpt2::Tensor bias({2});
            static_cast<void>(
                gpt2::quantized_linear(input, weight, scale, bias)
            );
        },
        "quantized linear rejects a non-matrix input"
    );

    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Tensor input({1, 3});
            const gpt2::Int8Tensor weight({3}, {1, 2, 3});
            const gpt2::Tensor scale({1}, std::vector<float>{1.0F});
            const gpt2::Tensor bias({2});
            static_cast<void>(
                gpt2::quantized_linear(input, weight, scale, bias)
            );
        },
        "quantized linear rejects a non-matrix weight"
    );

    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Tensor input({1, 3});
            const gpt2::Int8Tensor weight({3, 2}, {1, 2, 3, 4, 5, 6});
            const gpt2::Tensor scale({2}, std::vector<float>{1.0F, 1.0F});
            const gpt2::Tensor bias({1, 2});
            static_cast<void>(
                gpt2::quantized_linear(input, weight, scale, bias)
            );
        },
        "quantized linear rejects a non-vector bias"
    );
}

void test_invalid_quantized_linear_dimensions() {
    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Tensor input({2, 3});
            const gpt2::Int8Tensor weight(
                {4, 2},
                std::vector<std::int8_t>{1, 2, 3, 4, 5, 6, 7, 8}
            );
            const gpt2::Tensor scale({2}, std::vector<float>{1.0F, 1.0F});
            const gpt2::Tensor bias({2});
            static_cast<void>(
                gpt2::quantized_linear(input, weight, scale, bias)
            );
        },
        "quantized linear rejects an incompatible weight input size"
    );

    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Tensor input({2, 3});
            const gpt2::Int8Tensor weight(
                {3, 2},
                std::vector<std::int8_t>{1, 2, 3, 4, 5, 6}
            );
            const gpt2::Tensor scale({2}, std::vector<float>{1.0F, 1.0F});
            const gpt2::Tensor bias({3});
            static_cast<void>(
                gpt2::quantized_linear(input, weight, scale, bias)
            );
        },
        "quantized linear rejects an incompatible bias size"
    );

    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Tensor input({2, 3});
            const gpt2::Int8Tensor weight(
                {3, 2},
                std::vector<std::int8_t>{1, 2, 3, 4, 5, 6}
            );
            const gpt2::Tensor scale({3}, std::vector<float>(3, 1.0F));
            const gpt2::Tensor bias({2});
            static_cast<void>(
                gpt2::quantized_linear(input, weight, scale, bias)
            );
        },
        "quantized linear rejects a scale with the wrong length"
    );
}

}  // namespace

int main() {
    test_embedding_lookup();
    test_invalid_embedding_table_rank();
    test_empty_token_sequence();
    test_out_of_range_token_id();
    test_quantized_embedding_lookup();
    test_quantized_embedding_lookup_uses_each_rows_own_scale();
    test_invalid_quantized_embedding_table_rank();
    test_quantized_embedding_empty_token_sequence();
    test_quantized_embedding_out_of_range_token_id();
    test_quantized_embedding_scale_shape_mismatch();
    test_layer_norm_normalizes_rows_independently();
    test_layer_norm_applies_weight_and_bias();
    test_layer_norm_constant_row();
    test_invalid_layer_norm_parameters();
    test_invalid_layer_norm_epsilon();
    test_gelu_known_values();
    test_gelu_multidimensional_tensor();
    test_linear_transformation();
    test_linear_single_output_feature();
    test_invalid_linear_tensor_ranks();
    test_invalid_linear_dimensions();
    test_quantized_linear_transformation();
    test_quantized_linear_matches_dequantize_then_linear();
    test_invalid_quantized_linear_tensor_ranks();
    test_invalid_quantized_linear_dimensions();

    if (failure_count != 0) {
        std::cerr << failure_count << " test assertion(s) failed\n";
        return EXIT_FAILURE;
    }

    std::cout << "All layer tests passed\n";
    return EXIT_SUCCESS;
}
