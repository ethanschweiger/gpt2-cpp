#include "gpt2/tensor.h"

#include <array>
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
    const gpt2::Int8Tensor tensor(
        {2, 3},
        std::vector<std::int8_t>{0, 1, -2, 3, -4, 127}
    );

    expect(tensor.rank() == 2, "rank matches the number of dimensions");
    expect(tensor.numel() == 6, "numel is the product of the dimensions");
    expect(
        tensor.shape() == gpt2::Int8Tensor::Shape{2, 3},
        "shape preserves its dimensions"
    );
    expect(tensor.data()[4] == -4, "values constructor preserves data");
    expect(
        tensor.data()[5] == 127,
        "the maximum symmetric-quantization value round trips"
    );
}

void test_mutation() {
    gpt2::Int8Tensor tensor(
        {2, 2},
        std::vector<std::int8_t>{1, -1, 2, -2}
    );

    tensor.data()[0] = 100;
    const gpt2::Int8Tensor& const_tensor = tensor;
    expect(
        const_tensor.data()[0] == 100,
        "data exposes contiguous mutable storage"
    );
    expect(
        const_tensor.data()[3] == -2,
        "const data exposes stored values"
    );
}

void test_invalid_construction() {
    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Int8Tensor tensor(
                gpt2::Int8Tensor::Shape{},
                {}
            );
            static_cast<void>(tensor);
        },
        "an empty shape is rejected"
    );

    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Int8Tensor tensor(
                {2, 0, 3},
                {}
            );
            static_cast<void>(tensor);
        },
        "a zero-sized dimension is rejected"
    );

    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Int8Tensor tensor(
                {2, 2},
                std::vector<std::int8_t>{1, 2, 3}
            );
            static_cast<void>(tensor);
        },
        "a data-size mismatch is rejected"
    );

    expect_throws<std::overflow_error>(
        [] {
            const gpt2::Int8Tensor tensor(
                {std::numeric_limits<std::size_t>::max(), 2},
                {}
            );
            static_cast<void>(tensor);
        },
        "shape element-count overflow is rejected"
    );

    // -128 has no symmetric counterpart: round(x / scale) for a
    // symmetric scale (max(|W|) / 127) never produces it, so a -128
    // byte in a checkpoint always means something else went wrong.
    expect_throws<std::invalid_argument>(
        [] {
            const gpt2::Int8Tensor tensor(
                {2},
                std::vector<std::int8_t>{
                    0, std::numeric_limits<std::int8_t>::min()
                }
            );
            static_cast<void>(tensor);
        },
        "a value of -128 is rejected"
    );
}

}  // namespace

int main() {
    test_construction();
    test_mutation();
    test_invalid_construction();

    if (failure_count != 0) {
        std::cerr << failure_count << " int8 tensor test(s) failed\n";
        return EXIT_FAILURE;
    }

    std::cout << "all int8 tensor tests passed\n";
    return EXIT_SUCCESS;
}
