#include "gpt2/model.h"

#include "model_fixture.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

using gpt2_test::find_tensor;
using gpt2_test::make_checkpoint;
using gpt2_test::make_model_tensors;
using gpt2_test::make_tensor;
using gpt2_test::TemporaryCheckpoint;
using gpt2_test::TensorFixture;
using gpt2_test::ValueKind;

constexpr std::uint32_t vocabulary_size =
    gpt2_test::fixture_vocabulary_size;

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

gpt2::Gpt2Model load_model(
    const std::vector<TensorFixture>& tensors
) {
    const TemporaryCheckpoint file(make_checkpoint(tensors));
    return gpt2::Gpt2Model(gpt2::load_checkpoint(file.path()));
}

void test_complete_forward_matches_hugging_face() {
    TemporaryCheckpoint file(
        make_checkpoint(make_model_tensors())
    );
    gpt2::Gpt2Model model(gpt2::load_checkpoint(file.path()));

    expect(
        model.config().vocab_size == vocabulary_size,
        "model exposes its checkpoint configuration"
    );

    file.remove();
    expect(
        !std::filesystem::exists(file.path()),
        "test checkpoint is removed before inference"
    );

    const std::array<std::size_t, 3> token_ids{2, 5, 1};
    const gpt2::Tensor logits = model.forward(token_ids);

    expect(
        logits.shape() == gpt2::Tensor::Shape{3, 7},
        "complete model returns one vocabulary row per input token"
    );

    const std::array<float, 21> expected{
        -0.226207554F, 0.135856345F, 0.598779023F,
        -0.226193503F, 0.135870367F, -0.226263613F,
        -0.226179481F,

        0.510821521F, -0.539808154F, -0.233919472F,
        0.520559788F, -0.530069888F, 0.471868217F,
        0.530298114F,

        -0.366638601F, 0.665413082F, 0.047376595F,
        -0.374848485F, 0.657203197F, -0.333799064F,
        -0.383058369F,
    };

    for (std::size_t index = 0; index < expected.size(); ++index) {
        expect_near(
            logits.at(index),
            expected[index],
            2.0e-5F,
            "complete model matches Hugging Face FP32 logits"
        );
    }
}

void test_model_rejects_invalid_checkpoint_schema() {
    std::vector<TensorFixture> renamed = make_model_tensors();
    find_tensor(
        renamed,
        "transformer.h.0.ln_2.bias"
    ).name = "transformer.h.0.missing.weight";
    expect_throws<std::invalid_argument>(
        [&renamed] {
            static_cast<void>(load_model(renamed));
        },
        "model rejects a missing required tensor"
    );

    std::vector<TensorFixture> unexpected = make_model_tensors();
    unexpected.push_back(make_tensor(
        "lm_head.weight",
        {7, 4},
        ValueKind::weight,
        29
    ));
    expect_throws<std::invalid_argument>(
        [&unexpected] {
            static_cast<void>(load_model(unexpected));
        },
        "model rejects an unexpected language-model head tensor"
    );

    std::vector<TensorFixture> transposed_qkv = make_model_tensors();
    find_tensor(
        transposed_qkv,
        "transformer.h.0.attn.c_attn.weight"
    ).shape = {12, 4};
    expect_throws<std::invalid_argument>(
        [&transposed_qkv] {
            static_cast<void>(load_model(transposed_qkv));
        },
        "model rejects a QKV matrix with transposed dimensions"
    );
}

void test_model_rejects_invalid_token_sequences() {
    gpt2::Gpt2Model model = load_model(make_model_tensors());

    const std::array<std::size_t, 5> full_context{0, 1, 2, 3, 4};
    const gpt2::Tensor full_context_logits = model.forward(full_context);
    expect(
        full_context_logits.shape() == gpt2::Tensor::Shape{5, 7},
        "model accepts a sequence exactly as long as its context window"
    );

    const std::span<const std::size_t> empty;
    expect_throws<std::invalid_argument>(
        [&model, empty] {
            static_cast<void>(model.forward(empty));
        },
        "model rejects an empty token sequence"
    );

    const std::array<std::size_t, 6> too_long{0, 1, 2, 3, 4, 5};
    expect_throws<std::invalid_argument>(
        [&model, &too_long] {
            static_cast<void>(model.forward(too_long));
        },
        "model rejects a sequence longer than its context window"
    );

    const std::array<std::size_t, 1> invalid_token{7};
    expect_throws<std::out_of_range>(
        [&model, &invalid_token] {
            static_cast<void>(model.forward(invalid_token));
        },
        "model rejects a token outside its vocabulary"
    );
}

}  // namespace

int main() {
    test_complete_forward_matches_hugging_face();
    test_model_rejects_invalid_checkpoint_schema();
    test_model_rejects_invalid_token_sequences();

    if (failure_count != 0) {
        std::cerr << failure_count << " model test assertion(s) failed\n";
        return EXIT_FAILURE;
    }

    std::cout << "All model tests passed\n";
    return EXIT_SUCCESS;
}
