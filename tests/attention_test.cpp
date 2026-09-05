#include "gpt2/attention.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <utility>
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

void test_causal_mask_blocks_future_values() {
    const gpt2::Tensor query({3, 2});
    const gpt2::Tensor key({3, 2});
    const gpt2::Tensor value(
        {3, 2},
        std::vector<float>{
            1.0F, 10.0F,
            3.0F, 30.0F,
            5.0F, 50.0F,
        }
    );

    const gpt2::Tensor result =
        gpt2::causal_scaled_dot_product_attention(
            query,
            key,
            value
        );

    expect(
        result.shape() == gpt2::Tensor::Shape{3, 2},
        "causal attention preserves the query shape"
    );

    const std::array<float, 6> expected{
        1.0F, 10.0F,
        2.0F, 20.0F,
        3.0F, 30.0F,
    };

    for (std::size_t index = 0; index < expected.size(); ++index) {
        expect_near(
            result.at(index),
            expected[index],
            1.0e-5F,
            "causal attention uses only current and previous values"
        );
    }

    expect(query.at(0) == 0.0F, "attention leaves query unchanged");
    expect(key.at(0) == 0.0F, "attention leaves key unchanged");
    expect(value.at(0) == 1.0F, "attention leaves value unchanged");
}

void test_attention_scales_scores() {
    const gpt2::Tensor query(
        {2, 2},
        std::vector<float>{1.0F, 0.0F, 0.0F, 1.0F}
    );
    const gpt2::Tensor key(
        {2, 2},
        std::vector<float>{1.0F, 0.0F, 0.0F, 1.0F}
    );
    const gpt2::Tensor value(
        {2, 2},
        std::vector<float>{10.0F, 0.0F, 0.0F, 20.0F}
    );

    const gpt2::Tensor result =
        gpt2::causal_scaled_dot_product_attention(
            query,
            key,
            value
        );

    const std::array<float, 4> expected{
        10.0F,
        0.0F,
        3.30238F,
        13.3952F,
    };

    for (std::size_t index = 0; index < expected.size(); ++index) {
        expect_near(
            result.at(index),
            expected[index],
            1.0e-4F,
            "attention applies inverse-square-root head scaling"
        );
    }
}

void test_attention_rejects_invalid_ranks() {
    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Tensor query({2});
            const gpt2::Tensor key({1, 2});
            const gpt2::Tensor value({1, 2});
            static_cast<void>(
                gpt2::causal_scaled_dot_product_attention(
                    query,
                    key,
                    value
                )
            );
        },
        "attention rejects a non-matrix query"
    );

    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Tensor query({1, 2});
            const gpt2::Tensor key({1, 1, 2});
            const gpt2::Tensor value({1, 2});
            static_cast<void>(
                gpt2::causal_scaled_dot_product_attention(
                    query,
                    key,
                    value
                )
            );
        },
        "attention rejects a non-matrix key"
    );

    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Tensor query({1, 2});
            const gpt2::Tensor key({1, 2});
            const gpt2::Tensor value({1, 1, 2});
            static_cast<void>(
                gpt2::causal_scaled_dot_product_attention(
                    query,
                    key,
                    value
                )
            );
        },
        "attention rejects a non-matrix value"
    );
}

void test_attention_rejects_mismatched_shapes() {
    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Tensor query({2, 2});
            const gpt2::Tensor key({3, 2});
            const gpt2::Tensor value({2, 2});
            static_cast<void>(
                gpt2::causal_scaled_dot_product_attention(
                    query,
                    key,
                    value
                )
            );
        },
        "attention rejects mismatched query and key shapes"
    );

    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Tensor query({2, 2});
            const gpt2::Tensor key({2, 2});
            const gpt2::Tensor value({2, 3});
            static_cast<void>(
                gpt2::causal_scaled_dot_product_attention(
                    query,
                    key,
                    value
                )
            );
        },
        "attention rejects a mismatched value shape"
    );
}

void test_multi_head_attention_matches_reference_layout() {
    const gpt2::Tensor input(
        {2, 4},
        std::vector<float>{
            1.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 1.0F, 0.0F, 0.0F,
        }
    );
    const gpt2::Tensor qkv_weight(
        {4, 12},
        std::vector<float>{
            1.0F, 2.0F, 3.0F, 4.0F,
            0.0F, 0.0F, 0.0F, 2.0F,
            10.0F, 20.0F, 30.0F, 40.0F,

            1.0F, 0.0F, 0.0F, 1.0F,
            2.0F, 0.0F, 0.0F, 0.0F,
            50.0F, 60.0F, 70.0F, 80.0F,

            0.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 0.0F,

            0.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 0.0F,
        }
    );
    const gpt2::Tensor qkv_bias({12});
    const gpt2::Tensor output_weight(
        {4, 4},
        std::vector<float>{
            1.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 1.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 1.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 1.0F,
        }
    );
    const gpt2::Tensor output_bias({4});

    const gpt2::Tensor result =
        gpt2::multi_head_self_attention(
            input,
            qkv_weight,
            qkv_bias,
            output_weight,
            output_bias,
            2
        );

    expect(
        result.shape() == gpt2::Tensor::Shape{2, 4},
        "multi-head attention preserves the input shape"
    );

    const std::array<float, 8> expected{
        10.0F, 20.0F, 30.0F, 40.0F,
        42.17719F, 52.17719F, 37.82281F, 47.82281F,
    };

    for (std::size_t index = 0; index < expected.size(); ++index) {
        expect_near(
            result.at(index),
            expected[index],
            1.0e-4F,
            "multi-head attention matches the reference result"
        );
    }

    expect(input.at(0) == 1.0F, "multi-head attention leaves input unchanged");
    expect(
        qkv_weight.at(0) == 1.0F,
        "multi-head attention leaves QKV weight unchanged"
    );
    expect(
        output_weight.at(0) == 1.0F,
        "multi-head attention leaves output weight unchanged"
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

struct QuantizedAttentionFixture {
    gpt2::Tensor input;
    gpt2::Int8Tensor qkv_weight;
    gpt2::Tensor qkv_scale;
    gpt2::Tensor qkv_bias;
    gpt2::Int8Tensor output_weight;
    gpt2::Tensor output_scale;
    gpt2::Tensor output_bias;
    gpt2::Tensor dequantized_qkv_weight;
    gpt2::Tensor dequantized_output_weight;
};

QuantizedAttentionFixture make_quantized_attention_fixture(
    std::size_t sequence_length,
    std::size_t embedding_size
) {
    const std::size_t qkv_size = embedding_size * 3;

    gpt2::Int8Tensor qkv_weight(
        {embedding_size, qkv_size},
        make_deterministic_int8_values(embedding_size * qkv_size, 1)
    );
    gpt2::Tensor qkv_scale(
        {qkv_size},
        make_deterministic_scale_values(qkv_size, 2)
    );
    gpt2::Int8Tensor output_weight(
        {embedding_size, embedding_size},
        make_deterministic_int8_values(embedding_size * embedding_size, 3)
    );
    gpt2::Tensor output_scale(
        {embedding_size},
        make_deterministic_scale_values(embedding_size, 4)
    );

    QuantizedAttentionFixture fixture{
        gpt2::Tensor(
            {sequence_length, embedding_size},
            make_deterministic_float_values(
                sequence_length * embedding_size, 0
            )
        ),
        std::move(qkv_weight),
        std::move(qkv_scale),
        gpt2::Tensor(
            {qkv_size},
            make_deterministic_float_values(qkv_size, 5)
        ),
        std::move(output_weight),
        std::move(output_scale),
        gpt2::Tensor(
            {embedding_size},
            make_deterministic_float_values(embedding_size, 6)
        ),
        gpt2::Tensor({embedding_size, qkv_size}),
        gpt2::Tensor({embedding_size, embedding_size}),
    };

    fixture.dequantized_qkv_weight =
        dequantize(fixture.qkv_weight, fixture.qkv_scale);
    fixture.dequantized_output_weight =
        dequantize(fixture.output_weight, fixture.output_scale);

    return fixture;
}

void test_quantized_multi_head_attention_matches_dequantized_reference() {
    const QuantizedAttentionFixture fixture =
        make_quantized_attention_fixture(3, 4);

    const gpt2::Tensor quantized_result =
        gpt2::quantized_multi_head_self_attention(
            fixture.input,
            fixture.qkv_weight,
            fixture.qkv_scale,
            fixture.qkv_bias,
            fixture.output_weight,
            fixture.output_scale,
            fixture.output_bias,
            2
        );
    const gpt2::Tensor plain_result = gpt2::multi_head_self_attention(
        fixture.input,
        fixture.dequantized_qkv_weight,
        fixture.qkv_bias,
        fixture.dequantized_output_weight,
        fixture.output_bias,
        2
    );

    expect(
        quantized_result.shape() == plain_result.shape(),
        "quantized multi-head attention matches the plain result's shape"
    );
    for (std::size_t index = 0; index < quantized_result.numel(); ++index) {
        expect_near(
            quantized_result.at(index),
            plain_result.at(index),
            1.0e-3F,
            "quantized multi-head attention matches multi-head "
            "attention over explicitly dequantized weights"
        );
    }
}

void test_quantized_multi_head_attention_cached_matches_uncached() {
    const QuantizedAttentionFixture fixture =
        make_quantized_attention_fixture(4, 4);

    const gpt2::Tensor uncached = gpt2::quantized_multi_head_self_attention(
        fixture.input,
        fixture.qkv_weight,
        fixture.qkv_scale,
        fixture.qkv_bias,
        fixture.output_weight,
        fixture.output_scale,
        fixture.output_bias,
        2
    );

    gpt2::AttentionCache cache(fixture.input.shape()[0], 4);
    gpt2::Tensor cached_result({fixture.input.shape()[0], 4});
    for (std::size_t position = 0;
         position < fixture.input.shape()[0];
         ++position) {
        gpt2::Tensor step({1, 4});
        for (std::size_t feature = 0; feature < 4; ++feature) {
            const std::array<std::size_t, 2> step_index{0, feature};
            const std::array<std::size_t, 2> source_index{
                position, feature
            };
            step.at(step_index) = fixture.input.at(source_index);
        }

        const gpt2::Tensor step_result =
            gpt2::quantized_multi_head_self_attention(
                step,
                fixture.qkv_weight,
                fixture.qkv_scale,
                fixture.qkv_bias,
                fixture.output_weight,
                fixture.output_scale,
                fixture.output_bias,
                2,
                cache
            );

        for (std::size_t feature = 0; feature < 4; ++feature) {
            const std::array<std::size_t, 2> destination_index{
                position, feature
            };
            cached_result.at(destination_index) = step_result.at(
                std::array<std::size_t, 2>{0, feature}
            );
        }
    }

    for (std::size_t index = 0; index < uncached.numel(); ++index) {
        expect(
            uncached.at(index) == cached_result.at(index),
            "quantized attention's cached, one-token-at-a-time path "
            "reproduces the uncached result exactly"
        );
    }
}

void expect_invalid_quantized_multi_head_attention(
    const gpt2::Tensor& input,
    const gpt2::Int8Tensor& qkv_weight,
    const gpt2::Tensor& qkv_scale,
    const gpt2::Tensor& qkv_bias,
    const gpt2::Int8Tensor& output_weight,
    const gpt2::Tensor& output_scale,
    const gpt2::Tensor& output_bias,
    std::size_t head_count,
    std::string_view message
) {
    expect_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(gpt2::quantized_multi_head_self_attention(
                input,
                qkv_weight,
                qkv_scale,
                qkv_bias,
                output_weight,
                output_scale,
                output_bias,
                head_count
            ));
        },
        message
    );
}

void test_quantized_multi_head_attention_rejects_invalid_configuration() {
    const gpt2::Tensor input({1, 2});
    const gpt2::Int8Tensor qkv_weight(
        {2, 6},
        make_deterministic_int8_values(12, 10)
    );
    const gpt2::Tensor qkv_scale(
        {6},
        make_deterministic_scale_values(6, 11)
    );
    const gpt2::Tensor qkv_bias({6});
    const gpt2::Int8Tensor output_weight(
        {2, 2},
        make_deterministic_int8_values(4, 12)
    );
    const gpt2::Tensor output_scale(
        {2},
        make_deterministic_scale_values(2, 13)
    );
    const gpt2::Tensor output_bias({2});

    expect_invalid_quantized_multi_head_attention(
        input,
        qkv_weight,
        qkv_scale,
        qkv_bias,
        output_weight,
        output_scale,
        output_bias,
        0,
        "quantized multi-head attention rejects zero heads"
    );

    const gpt2::Tensor bad_qkv_scale({5}, std::vector<float>(5, 1.0F));
    expect_invalid_quantized_multi_head_attention(
        input,
        qkv_weight,
        bad_qkv_scale,
        qkv_bias,
        output_weight,
        output_scale,
        output_bias,
        1,
        "quantized multi-head attention rejects an incorrect QKV scale "
        "size"
    );

    const gpt2::Tensor bad_output_scale({3}, std::vector<float>(3, 1.0F));
    expect_invalid_quantized_multi_head_attention(
        input,
        qkv_weight,
        qkv_scale,
        qkv_bias,
        output_weight,
        bad_output_scale,
        output_bias,
        1,
        "quantized multi-head attention rejects an incorrect output "
        "scale size"
    );
}

void test_multi_head_attention_applies_biases_and_output_projection() {
    const gpt2::Tensor input({1, 2});
    const gpt2::Tensor qkv_weight({2, 6});
    const gpt2::Tensor qkv_bias(
        {6},
        std::vector<float>{0.0F, 0.0F, 0.0F, 0.0F, 2.0F, 3.0F}
    );
    const gpt2::Tensor output_weight(
        {2, 2},
        std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F}
    );
    const gpt2::Tensor output_bias(
        {2},
        std::vector<float>{0.5F, -1.0F}
    );

    const gpt2::Tensor result =
        gpt2::multi_head_self_attention(
            input,
            qkv_weight,
            qkv_bias,
            output_weight,
            output_bias,
            2
        );

    expect_near(
        result.at(0),
        11.5F,
        1.0e-5F,
        "multi-head attention applies the output projection"
    );
    expect_near(
        result.at(1),
        15.0F,
        1.0e-5F,
        "multi-head attention applies QKV and output biases"
    );
}

void expect_invalid_multi_head_attention(
    const gpt2::Tensor& input,
    const gpt2::Tensor& qkv_weight,
    const gpt2::Tensor& qkv_bias,
    const gpt2::Tensor& output_weight,
    const gpt2::Tensor& output_bias,
    std::size_t head_count,
    std::string_view message
) {
    expect_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(gpt2::multi_head_self_attention(
                input,
                qkv_weight,
                qkv_bias,
                output_weight,
                output_bias,
                head_count
            ));
        },
        message
    );
}

void test_multi_head_attention_rejects_invalid_configuration() {
    const gpt2::Tensor input({1, 2});
    const gpt2::Tensor qkv_weight({2, 6});
    const gpt2::Tensor qkv_bias({6});
    const gpt2::Tensor output_weight({2, 2});
    const gpt2::Tensor output_bias({2});

    expect_invalid_multi_head_attention(
        input,
        qkv_weight,
        qkv_bias,
        output_weight,
        output_bias,
        0,
        "multi-head attention rejects zero heads"
    );

    const gpt2::Tensor vector_input({2});
    expect_invalid_multi_head_attention(
        vector_input,
        qkv_weight,
        qkv_bias,
        output_weight,
        output_bias,
        1,
        "multi-head attention rejects a non-matrix input"
    );

    const gpt2::Tensor three_feature_input({1, 3});
    const gpt2::Tensor three_feature_qkv_weight({3, 9});
    const gpt2::Tensor three_feature_qkv_bias({9});
    const gpt2::Tensor three_feature_output_weight({3, 3});
    const gpt2::Tensor three_feature_output_bias({3});
    expect_invalid_multi_head_attention(
        three_feature_input,
        three_feature_qkv_weight,
        three_feature_qkv_bias,
        three_feature_output_weight,
        three_feature_output_bias,
        2,
        "multi-head attention rejects a nondivisible head count"
    );

    const gpt2::Tensor bad_qkv_weight({2, 5});
    expect_invalid_multi_head_attention(
        input,
        bad_qkv_weight,
        qkv_bias,
        output_weight,
        output_bias,
        1,
        "multi-head attention rejects an incorrect QKV weight shape"
    );

    const gpt2::Tensor bad_qkv_bias({5});
    expect_invalid_multi_head_attention(
        input,
        qkv_weight,
        bad_qkv_bias,
        output_weight,
        output_bias,
        1,
        "multi-head attention rejects an incorrect QKV bias size"
    );

    const gpt2::Tensor matrix_qkv_bias({1, 6});
    expect_invalid_multi_head_attention(
        input,
        qkv_weight,
        matrix_qkv_bias,
        output_weight,
        output_bias,
        1,
        "multi-head attention rejects a non-vector QKV bias"
    );

    const gpt2::Tensor bad_output_weight({2, 3});
    expect_invalid_multi_head_attention(
        input,
        qkv_weight,
        qkv_bias,
        bad_output_weight,
        output_bias,
        1,
        "multi-head attention rejects an incorrect output weight shape"
    );

    const gpt2::Tensor bad_output_bias({3});
    expect_invalid_multi_head_attention(
        input,
        qkv_weight,
        qkv_bias,
        output_weight,
        bad_output_bias,
        1,
        "multi-head attention rejects an incorrect output bias size"
    );

    const gpt2::Tensor matrix_output_bias({1, 2});
    expect_invalid_multi_head_attention(
        input,
        qkv_weight,
        qkv_bias,
        output_weight,
        matrix_output_bias,
        1,
        "multi-head attention rejects a non-vector output bias"
    );
}

}  // namespace

int main() {
    test_causal_mask_blocks_future_values();
    test_attention_scales_scores();
    test_attention_rejects_invalid_ranks();
    test_attention_rejects_mismatched_shapes();
    test_multi_head_attention_matches_reference_layout();
    test_multi_head_attention_applies_biases_and_output_projection();
    test_multi_head_attention_rejects_invalid_configuration();
    test_quantized_multi_head_attention_matches_dequantized_reference();
    test_quantized_multi_head_attention_cached_matches_uncached();
    test_quantized_multi_head_attention_rejects_invalid_configuration();

    if (failure_count != 0) {
        std::cerr << failure_count << " attention test assertion(s) failed\n";
        return EXIT_FAILURE;
    }

    std::cout << "All attention tests passed\n";
    return EXIT_SUCCESS;
}
