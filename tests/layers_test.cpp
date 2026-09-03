#include "gpt2/layers.h"

#include <array>
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

}  // namespace

int main() {
    test_embedding_lookup();
    test_invalid_embedding_table_rank();
    test_empty_token_sequence();
    test_out_of_range_token_id();

    if (failure_count != 0) {
        std::cerr << failure_count << " test assertion(s) failed\n";
        return EXIT_FAILURE;
    }

    std::cout << "All layer tests passed\n";
    return EXIT_SUCCESS;
}
