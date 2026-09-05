#include "gpt2/tensor_ops.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

void test_one_dimensional_addition() {
    const gpt2::Tensor left(
        {4},
        std::vector<float>{1.0F, -2.0F, 3.5F, 0.0F}
    );
    const gpt2::Tensor right(
        {4},
        std::vector<float>{4.0F, 2.0F, -1.5F, 8.0F}
    );

    const gpt2::Tensor result = gpt2::add(left, right);

    expect(result.shape() == gpt2::Tensor::Shape{4}, "1D shape is preserved");
    expect(result.at(0) == 5.0F, "1D sum at index 0 is correct");
    expect(result.at(1) == 0.0F, "1D sum at index 1 is correct");
    expect(result.at(2) == 2.0F, "1D sum at index 2 is correct");
    expect(result.at(3) == 8.0F, "1D sum at index 3 is correct");
}

void test_two_dimensional_addition() {
    const gpt2::Tensor left(
        {2, 2},
        std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F}
    );
    const gpt2::Tensor right(
        {2, 2},
        std::vector<float>{0.5F, 1.5F, 2.5F, 3.5F}
    );

    const gpt2::Tensor result = gpt2::add(left, right);

    expect(
        result.shape() == gpt2::Tensor::Shape{2, 2},
        "2D shape is preserved"
    );
    expect(result.at(0) == 1.5F, "2D sum at flat index 0 is correct");
    expect(result.at(1) == 3.5F, "2D sum at flat index 1 is correct");
    expect(result.at(2) == 5.5F, "2D sum at flat index 2 is correct");
    expect(result.at(3) == 7.5F, "2D sum at flat index 3 is correct");

    expect(left.at(0) == 1.0F, "left input is unchanged");
    expect(right.at(3) == 3.5F, "right input is unchanged");
}

void test_shape_mismatch() {
    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Tensor left({2, 3});
            const gpt2::Tensor right({3, 2});
            static_cast<void>(gpt2::add(left, right));
        },
        "different shapes with equal element counts are rejected"
    );

    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Tensor left({2});
            const gpt2::Tensor right({3});
            static_cast<void>(gpt2::add(left, right));
        },
        "different shapes with different element counts are rejected"
    );
}

void test_matrix_multiplication() {
    const gpt2::Tensor left(
        {2, 3},
        std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}
    );
    const gpt2::Tensor right(
        {3, 2},
        std::vector<float>{7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F}
    );

    const gpt2::Tensor result = gpt2::matmul(left, right);

    expect(
        result.shape() == gpt2::Tensor::Shape{2, 2},
        "matrix multiplication produces the expected shape"
    );
    expect(result.at(0) == 58.0F, "matrix result at row 0 column 0 is correct");
    expect(result.at(1) == 64.0F, "matrix result at row 0 column 1 is correct");
    expect(result.at(2) == 139.0F, "matrix result at row 1 column 0 is correct");
    expect(result.at(3) == 154.0F, "matrix result at row 1 column 1 is correct");

    expect(left.at(0) == 1.0F, "matmul leaves the left input unchanged");
    expect(right.at(5) == 12.0F, "matmul leaves the right input unchanged");
}

void test_matrix_by_column_vector() {
    const gpt2::Tensor matrix(
        {2, 3},
        std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}
    );
    const gpt2::Tensor column(
        {3, 1},
        std::vector<float>{1.0F, 2.0F, 3.0F}
    );

    const gpt2::Tensor result = gpt2::matmul(matrix, column);

    expect(
        result.shape() == gpt2::Tensor::Shape{2, 1},
        "matrix-by-column-vector produces the expected shape"
    );
    expect(result.at(0) == 14.0F, "first matrix-vector result is correct");
    expect(result.at(1) == 32.0F, "second matrix-vector result is correct");
}

void test_identity_matrix_multiplication() {
    const gpt2::Tensor matrix(
        {2, 2},
        std::vector<float>{-1.0F, 2.5F, 3.0F, 4.0F}
    );
    const gpt2::Tensor identity(
        {2, 2},
        std::vector<float>{1.0F, 0.0F, 0.0F, 1.0F}
    );

    const gpt2::Tensor result = gpt2::matmul(matrix, identity);

    for (std::size_t index = 0; index < matrix.numel(); ++index) {
        expect(
            result.at(index) == matrix.at(index),
            "multiplication by the identity matrix preserves values"
        );
    }
}

void test_invalid_matrix_multiplication() {
    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Tensor left({3});
            const gpt2::Tensor right({3, 1});
            static_cast<void>(gpt2::matmul(left, right));
        },
        "matmul rejects a left operand that is not rank 2"
    );

    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Tensor left({1, 2});
            const gpt2::Tensor right({2, 1, 1});
            static_cast<void>(gpt2::matmul(left, right));
        },
        "matmul rejects a right operand that is not rank 2"
    );

    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Tensor left({2, 3});
            const gpt2::Tensor right({2, 2});
            static_cast<void>(gpt2::matmul(left, right));
        },
        "matmul rejects incompatible inner dimensions"
    );
}

void test_quantized_matrix_multiplication() {
    const gpt2::Tensor left(
        {2, 3},
        std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}
    );
    // Values chosen as exact multiples of their column's scale, so the
    // dequantized weight -- and therefore the expected result -- has
    // no rounding to reason about.
    const gpt2::Int8Tensor right(
        {3, 2},
        std::vector<std::int8_t>{10, 20, 30, 40, 50, 60}
    );
    const gpt2::Tensor scale({2}, std::vector<float>{1.0F, 2.0F});

    const gpt2::Tensor result = gpt2::quantized_matmul(left, right, scale);

    expect(
        result.shape() == gpt2::Tensor::Shape{2, 2},
        "quantized matrix multiplication produces the expected shape"
    );
    expect(
        result.at(0) == 220.0F,
        "quantized result at row 0 column 0 is correct"
    );
    expect(
        result.at(1) == 560.0F,
        "quantized result at row 0 column 1 is correct"
    );
    expect(
        result.at(2) == 490.0F,
        "quantized result at row 1 column 0 is correct"
    );
    expect(
        result.at(3) == 1280.0F,
        "quantized result at row 1 column 1 is correct"
    );

    expect(left.at(0) == 1.0F, "quantized_matmul leaves the input unchanged");
    expect(
        right.data()[0] == 10,
        "quantized_matmul leaves the quantized weight unchanged"
    );
}

void test_quantized_matmul_matches_dequantize_then_matmul() {
    const gpt2::Tensor left(
        {2, 4},
        std::vector<float>{
            -3.5F, 2.0F, 0.25F, 7.0F,
            1.0F, -6.0F, 4.5F, 0.5F
        }
    );
    const std::vector<std::int8_t> quantized_values{
        127, -64, 0, 50,
        -100, 30, 90, -127,
        10, 10, -10, 20
    };
    const gpt2::Int8Tensor right({4, 3}, quantized_values);
    const gpt2::Tensor scale(
        {3},
        std::vector<float>{0.1F, 0.02F, 0.5F}
    );

    gpt2::Tensor dequantized_right({4, 3});
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            const std::array<std::size_t, 2> index{row, column};
            dequantized_right.at(index) =
                static_cast<float>(quantized_values[row * 3 + column]) *
                scale.at(column);
        }
    }

    const gpt2::Tensor quantized_result =
        gpt2::quantized_matmul(left, right, scale);
    const gpt2::Tensor plain_result = gpt2::matmul(left, dequantized_right);

    for (std::size_t index = 0; index < quantized_result.numel(); ++index) {
        expect_near(
            quantized_result.at(index),
            plain_result.at(index),
            1.0e-4F,
            "quantized_matmul matches matmul over an explicitly "
            "dequantized copy of the same weight"
        );
    }
}

void test_invalid_quantized_matrix_multiplication() {
    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Tensor left({3});
            const gpt2::Int8Tensor right({3, 1}, {1, 2, 3});
            const gpt2::Tensor scale({1}, std::vector<float>{1.0F});
            static_cast<void>(gpt2::quantized_matmul(left, right, scale));
        },
        "quantized_matmul rejects a left operand that is not rank 2"
    );

    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Tensor left({2, 3});
            const gpt2::Int8Tensor right({2, 2}, {1, 2, 3, 4});
            const gpt2::Tensor scale({2}, std::vector<float>{1.0F, 1.0F});
            static_cast<void>(gpt2::quantized_matmul(left, right, scale));
        },
        "quantized_matmul rejects incompatible inner dimensions"
    );

    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Tensor left({2, 3});
            const gpt2::Int8Tensor right(
                {3, 2},
                std::vector<std::int8_t>{1, 2, 3, 4, 5, 6}
            );
            const gpt2::Tensor scale(
                {3},
                std::vector<float>{1.0F, 1.0F, 1.0F}
            );
            static_cast<void>(gpt2::quantized_matmul(left, right, scale));
        },
        "quantized_matmul rejects a scale with the wrong length"
    );

    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Tensor left({2, 3});
            const gpt2::Int8Tensor right(
                {3, 2},
                std::vector<std::int8_t>{1, 2, 3, 4, 5, 6}
            );
            const gpt2::Tensor scale({1, 2}, std::vector<float>{1.0F, 1.0F});
            static_cast<void>(gpt2::quantized_matmul(left, right, scale));
        },
        "quantized_matmul rejects a scale that is not rank 1"
    );
}

void test_matrix_transpose() {
    const gpt2::Tensor input(
        {2, 3},
        std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}
    );

    const gpt2::Tensor result = gpt2::transpose(input);

    expect(
        result.shape() == gpt2::Tensor::Shape{3, 2},
        "transpose swaps the matrix dimensions"
    );

    const std::array<float, 6> expected{
        1.0F, 4.0F,
        2.0F, 5.0F,
        3.0F, 6.0F,
    };

    for (std::size_t index = 0; index < expected.size(); ++index) {
        expect(
            result.at(index) == expected[index],
            "transpose moves each value to the expected position"
        );
    }

    expect(input.at(1) == 2.0F, "transpose leaves its input unchanged");
}

void test_invalid_matrix_transpose() {
    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Tensor input({3});
            static_cast<void>(gpt2::transpose(input));
        },
        "transpose rejects a rank-1 tensor"
    );

    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Tensor input({1, 2, 3});
            static_cast<void>(gpt2::transpose(input));
        },
        "transpose rejects a rank-3 tensor"
    );
}

void test_stable_row_wise_softmax() {
    const gpt2::Tensor input(
        {2, 3},
        std::vector<float>{
            1.0F, 2.0F, 3.0F,
            1000.0F, 1001.0F, 1002.0F,
        }
    );

    const gpt2::Tensor result = gpt2::softmax(input);

    expect(
        result.shape() == input.shape(),
        "softmax preserves the input shape"
    );

    const std::array<float, 3> expected{
        0.0900306F,
        0.244728F,
        0.665241F,
    };

    for (std::size_t row = 0; row < 2; ++row) {
        float row_sum = 0.0F;

        for (std::size_t column = 0; column < 3; ++column) {
            const std::size_t index = row * 3 + column;

            expect_near(
                result.at(index),
                expected[column],
                1.0e-5F,
                "softmax produces the expected probability"
            );
            row_sum += result.at(index);
        }

        expect_near(
            row_sum,
            1.0F,
            1.0e-6F,
            "each softmax row sums to one"
        );
    }

    expect(
        input.at(3) == 1000.0F,
        "softmax leaves its input unchanged"
    );
}

void test_softmax_uses_final_dimension() {
    const gpt2::Tensor input(
        {2, 2, 2},
        std::vector<float>{
            1.0F, 1.0F,
            2.0F, 2.0F,
            100.0F, 100.0F,
            -5.0F, -5.0F,
        }
    );

    const gpt2::Tensor result = gpt2::softmax(input);

    expect(
        result.shape() == input.shape(),
        "softmax preserves a rank-3 shape"
    );

    for (std::size_t index = 0; index < result.numel(); ++index) {
        expect_near(
            result.at(index),
            0.5F,
            1.0e-6F,
            "softmax normalizes each final-dimension pair independently"
        );
    }
}

}  // namespace

int main() {
    test_one_dimensional_addition();
    test_two_dimensional_addition();
    test_shape_mismatch();
    test_matrix_multiplication();
    test_matrix_by_column_vector();
    test_identity_matrix_multiplication();
    test_invalid_matrix_multiplication();
    test_quantized_matrix_multiplication();
    test_quantized_matmul_matches_dequantize_then_matmul();
    test_invalid_quantized_matrix_multiplication();
    test_matrix_transpose();
    test_invalid_matrix_transpose();
    test_stable_row_wise_softmax();
    test_softmax_uses_final_dimension();

    if (failure_count != 0) {
        std::cerr << failure_count << " tensor operation test(s) failed\n";
        return EXIT_FAILURE;
    }

    std::cout << "all tensor operation tests passed\n";
    return EXIT_SUCCESS;
}
