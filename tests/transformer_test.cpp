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

}  // namespace

int main() {
    test_feed_forward();
    test_zero_sublayers_preserve_residual();
    test_transformer_block_matches_hugging_face_reference();
    test_transformer_block_rejects_invalid_input();

    if (failure_count != 0) {
        std::cerr << failure_count << " transformer test assertion(s) failed\n";
        return EXIT_FAILURE;
    }

    std::cout << "All transformer tests passed\n";
    return EXIT_SUCCESS;
}
