#include "gpt2/attention.h"

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

}  // namespace

int main() {
    test_causal_mask_blocks_future_values();
    test_attention_scales_scores();
    test_attention_rejects_invalid_ranks();
    test_attention_rejects_mismatched_shapes();

    if (failure_count != 0) {
        std::cerr << failure_count << " attention test assertion(s) failed\n";
        return EXIT_FAILURE;
    }

    std::cout << "All attention tests passed\n";
    return EXIT_SUCCESS;
}
