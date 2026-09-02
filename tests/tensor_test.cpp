#include "gpt2/tensor.h"

#include <array>
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

void test_construction() {
    const gpt2::Tensor zeros({2, 3});

    expect(zeros.rank() == 2, "rank matches the number of dimensions");
    expect(zeros.numel() == 6, "numel is the product of the dimensions");
    expect(
        zeros.shape() == gpt2::Tensor::Shape{2, 3},
        "shape preserves its dimensions"
    );

    for (std::size_t index = 0; index < zeros.numel(); ++index) {
        expect(zeros.at(index) == 0.0F, "default fill value is zero");
    }

    const gpt2::Tensor filled({2, 2}, 3.5F);
    for (std::size_t index = 0; index < filled.numel(); ++index) {
        expect(filled.at(index) == 3.5F, "custom fill value is preserved");
    }

    const gpt2::Tensor values(
        {2, 3},
        std::vector<float>{0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F}
    );
    expect(values.at(4) == 4.0F, "values constructor preserves data");
}

void test_indexing_and_mutation() {
    gpt2::Tensor matrix(
        {2, 3},
        std::vector<float>{0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F}
    );

    const std::array<std::size_t, 2> matrix_index{1, 2};
    expect(matrix.at(matrix_index) == 5.0F, "2D indexing is row-major");

    matrix.at(matrix_index) = 42.0F;
    expect(matrix.at(5) == 42.0F, "coordinate mutation updates flat storage");

    matrix.data()[0] = -1.0F;
    const gpt2::Tensor& const_matrix = matrix;
    expect(const_matrix.at(0) == -1.0F, "data exposes contiguous mutable storage");
    expect(const_matrix.data()[5] == 42.0F, "const data exposes stored values");

    const gpt2::Tensor cube(
        {2, 3, 4},
        std::vector<float>{
            0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F,
            6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F,
            12.0F, 13.0F, 14.0F, 15.0F, 16.0F, 17.0F,
            18.0F, 19.0F, 20.0F, 21.0F, 22.0F, 23.0F
        }
    );
    const std::array<std::size_t, 3> cube_index{1, 2, 3};
    expect(cube.at(cube_index) == 23.0F, "3D indexing is row-major");
}

void test_invalid_construction() {
    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Tensor tensor(gpt2::Tensor::Shape{});
            static_cast<void>(tensor);
        },
        "an empty shape is rejected"
    );

    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Tensor tensor({2, 0, 3});
            static_cast<void>(tensor);
        },
        "a zero-sized dimension is rejected"
    );

    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Tensor tensor(
                {2, 2},
                std::vector<float>{1.0F, 2.0F, 3.0F}
            );
            static_cast<void>(tensor);
        },
        "a data-size mismatch is rejected"
    );

    expect_throws<std::overflow_error>(
        [] {
            const gpt2::Tensor tensor({
                std::numeric_limits<std::size_t>::max(),
                2
            });
            static_cast<void>(tensor);
        },
        "shape element-count overflow is rejected"
    );
}

void test_invalid_indexing() {
    const gpt2::Tensor tensor({2, 3});

    expect_throws<std::out_of_range>(
        [&tensor] { static_cast<void>(tensor.at(6)); },
        "a flat index past the end is rejected"
    );

    expect_throws<std::invalid_argument>(
        [&tensor] {
            const std::array<std::size_t, 1> index{1};
            static_cast<void>(tensor.at(index));
        },
        "an index with the wrong rank is rejected"
    );

    expect_throws<std::out_of_range>(
        [&tensor] {
            const std::array<std::size_t, 2> index{2, 0};
            static_cast<void>(tensor.at(index));
        },
        "an out-of-range coordinate is rejected"
    );
}

}  // namespace

int main() {
    test_construction();
    test_indexing_and_mutation();
    test_invalid_construction();
    test_invalid_indexing();

    if (failure_count != 0) {
        std::cerr << failure_count << " tensor test(s) failed\n";
        return EXIT_FAILURE;
    }

    std::cout << "all tensor tests passed\n";
    return EXIT_SUCCESS;
}
