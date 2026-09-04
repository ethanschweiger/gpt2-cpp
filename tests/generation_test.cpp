#include "gpt2/generation.h"

#include "model_fixture.h"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using gpt2_test::alternate_index_multiplier;
using gpt2_test::find_tensor;
using gpt2_test::make_checkpoint;
using gpt2_test::make_model_tensors;
using gpt2_test::TemporaryCheckpoint;
using gpt2_test::TensorFixture;

using TokenIds = std::vector<std::size_t>;

constexpr std::size_t fixture_context_length =
    gpt2_test::fixture_context_length;

int failure_count = 0;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failure_count;
    }
}

std::string describe(const TokenIds& token_ids) {
    std::ostringstream stream;
    stream << '[';
    for (std::size_t index = 0; index < token_ids.size(); ++index) {
        if (index != 0) {
            stream << ", ";
        }
        stream << token_ids[index];
    }
    stream << ']';
    return stream.str();
}

std::string_view describe(gpt2::GenerationStop stop) {
    switch (stop) {
    case gpt2::GenerationStop::token_limit:
        return "token_limit";
    case gpt2::GenerationStop::end_of_text:
        return "end_of_text";
    case gpt2::GenerationStop::context_limit:
        return "context_limit";
    }

    return "unknown";
}

void expect_generation(
    const gpt2::Generation& actual,
    const TokenIds& expected_tokens,
    gpt2::GenerationStop expected_stop,
    std::string_view message
) {
    if (actual.new_token_ids != expected_tokens) {
        std::cerr << "FAIL: " << message
                  << " (expected " << describe(expected_tokens)
                  << ", got " << describe(actual.new_token_ids)
                  << ")\n";
        ++failure_count;
    }

    if (actual.stop != expected_stop) {
        std::cerr << "FAIL: " << message
                  << " (expected stop " << describe(expected_stop)
                  << ", got " << describe(actual.stop) << ")\n";
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

gpt2::Gpt2Model load_model(const std::vector<TensorFixture>& tensors) {
    const TemporaryCheckpoint file(
        make_checkpoint(tensors),
        "gpt2-generation-test-"
    );
    return gpt2::Gpt2Model(gpt2::load_checkpoint(file.path()));
}

gpt2::Gpt2Model load_fixture_model() {
    return load_model(make_model_tensors(alternate_index_multiplier));
}

gpt2::GenerationLimits token_limit(std::size_t maximum) {
    gpt2::GenerationLimits options;
    options.maximum_new_tokens = maximum;
    return options;
}

// Every expectation below comes from running the same fixture weights
// through Hugging Face's GPT2LMHeadModel.generate(do_sample=False).
void test_greedy_generation_matches_hugging_face() {
    const gpt2::Gpt2Model model = load_fixture_model();

    const std::array<std::size_t, 3> prompt{2, 5, 1};
    expect_generation(
        gpt2::generate_greedy(model, prompt, token_limit(8)),
        {1, 2},
        gpt2::GenerationStop::context_limit,
        "greedy generation fills the context window"
    );

    const std::array<std::size_t, 1> short_prompt{1};
    expect_generation(
        gpt2::generate_greedy(model, short_prompt, token_limit(8)),
        {1, 1, 1, 2},
        gpt2::GenerationStop::context_limit,
        "greedy generation extends a one-token prompt"
    );

    const std::array<std::size_t, 2> pair_prompt{0, 3};
    expect_generation(
        gpt2::generate_greedy(model, pair_prompt, token_limit(8)),
        {1, 1, 2},
        gpt2::GenerationStop::context_limit,
        "greedy generation extends a two-token prompt"
    );

    const std::array<std::size_t, 1> repeating_prompt{2};
    expect_generation(
        gpt2::generate_greedy(model, repeating_prompt, token_limit(8)),
        {2, 2, 2, 2},
        gpt2::GenerationStop::context_limit,
        "greedy generation repeats a fixed point"
    );
}

// The continuation of {2, 5, 1} changes between its two steps, so a
// run that reused the first step's logits would fail these.
void test_each_step_re_runs_the_forward_pass() {
    const gpt2::Gpt2Model model = load_fixture_model();
    const std::array<std::size_t, 3> prompt{2, 5, 1};

    expect_generation(
        gpt2::generate_greedy(model, prompt, token_limit(1)),
        {1},
        gpt2::GenerationStop::token_limit,
        "one step takes the first token"
    );
    expect_generation(
        gpt2::generate_greedy(model, prompt, token_limit(2)),
        {1, 2},
        gpt2::GenerationStop::token_limit,
        "the second step differs from the first"
    );
}

void test_generation_respects_its_limits() {
    const gpt2::Gpt2Model model = load_fixture_model();
    const std::array<std::size_t, 1> prompt{1};

    expect_generation(
        gpt2::generate_greedy(model, prompt, token_limit(0)),
        {},
        gpt2::GenerationStop::token_limit,
        "a zero token limit generates nothing"
    );
    expect_generation(
        gpt2::generate_greedy(model, prompt, token_limit(2)),
        {1, 1},
        gpt2::GenerationStop::token_limit,
        "the token limit stops generation early"
    );

    const std::array<std::size_t, 5> full_context{0, 1, 2, 3, 4};
    expect_generation(
        gpt2::generate_greedy(model, full_context, token_limit(4)),
        {},
        gpt2::GenerationStop::context_limit,
        "a full prompt leaves no room to generate"
    );

    expect(
        fixture_context_length == 5,
        "the fixture context window is five tokens"
    );
}

void test_generation_stops_at_the_end_of_text_token() {
    const gpt2::Gpt2Model model = load_fixture_model();
    const std::array<std::size_t, 1> prompt{1};

    gpt2::GenerationLimits stop_on_first = token_limit(8);
    stop_on_first.end_of_text_id = 1;
    expect_generation(
        gpt2::generate_greedy(model, prompt, stop_on_first),
        {1},
        gpt2::GenerationStop::end_of_text,
        "generation stops on the end-of-text token"
    );

    gpt2::GenerationLimits stop_on_last = token_limit(8);
    stop_on_last.end_of_text_id = 2;
    expect_generation(
        gpt2::generate_greedy(model, prompt, stop_on_last),
        {1, 1, 1, 2},
        gpt2::GenerationStop::end_of_text,
        "generation stops on a later end-of-text token"
    );

    gpt2::GenerationLimits never_stops = token_limit(8);
    never_stops.end_of_text_id = 6;
    expect_generation(
        gpt2::generate_greedy(model, prompt, never_stops),
        {1, 1, 1, 2},
        gpt2::GenerationStop::context_limit,
        "an end-of-text token that never appears changes nothing"
    );
}

void test_generation_rejects_invalid_prompts() {
    const gpt2::Gpt2Model model = load_fixture_model();

    const std::span<const std::size_t> empty;
    expect_throws<std::invalid_argument>(
        [&model, empty] {
            static_cast<void>(
                gpt2::generate_greedy(model, empty, token_limit(1))
            );
        },
        "generation rejects an empty prompt"
    );

    const std::array<std::size_t, 6> too_long{0, 1, 2, 3, 4, 5};
    expect_throws<std::invalid_argument>(
        [&model, &too_long] {
            static_cast<void>(
                gpt2::generate_greedy(model, too_long, token_limit(1))
            );
        },
        "generation rejects a prompt longer than the context window"
    );

    const std::array<std::size_t, 1> invalid_token{7};
    expect_throws<std::out_of_range>(
        [&model, &invalid_token] {
            static_cast<void>(
                gpt2::generate_greedy(model, invalid_token, token_limit(1))
            );
        },
        "generation rejects a token outside the vocabulary"
    );
}

void test_generation_rejects_non_finite_logits() {
    std::vector<TensorFixture> tensors =
        make_model_tensors(alternate_index_multiplier);
    TensorFixture& final_scale =
        find_tensor(tensors, "transformer.ln_f.weight");
    final_scale.values[0] =
        std::bit_cast<float>(std::uint32_t{0x7FC00000});

    const gpt2::Gpt2Model model = load_model(tensors);
    const std::array<std::size_t, 1> prompt{1};
    expect_throws<std::runtime_error>(
        [&model, &prompt] {
            static_cast<void>(
                gpt2::generate_greedy(model, prompt, token_limit(1))
            );
        },
        "generation rejects a model that produces no finite logit"
    );
}

}  // namespace

int main() {
    try {
        test_greedy_generation_matches_hugging_face();
        test_each_step_re_runs_the_forward_pass();
        test_generation_respects_its_limits();
        test_generation_stops_at_the_end_of_text_token();
        test_generation_rejects_invalid_prompts();
        test_generation_rejects_non_finite_logits();
    } catch (const std::exception& exception) {
        std::cerr << "FAIL: unexpected exception: "
                  << exception.what() << '\n';
        return EXIT_FAILURE;
    }

    if (failure_count != 0) {
        std::cerr << failure_count
                  << " generation test assertion(s) failed\n";
        return EXIT_FAILURE;
    }

    std::cout << "All generation tests passed\n";
    return EXIT_SUCCESS;
}
