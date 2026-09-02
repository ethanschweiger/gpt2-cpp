#include "gpt2/tensor_ops.h"

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

}  // namespace

int main() {
    test_one_dimensional_addition();
    test_two_dimensional_addition();
    test_shape_mismatch();
    test_matrix_multiplication();
    test_matrix_by_column_vector();
    test_identity_matrix_multiplication();
    test_invalid_matrix_multiplication();

    if (failure_count != 0) {
        std::cerr << failure_count << " tensor operation test(s) failed\n";
        return EXIT_FAILURE;
    }

    std::cout << "all tensor operation tests passed\n";
    return EXIT_SUCCESS;
}
