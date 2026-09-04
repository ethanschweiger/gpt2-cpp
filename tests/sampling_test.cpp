#include "gpt2/generation.h"

#include "model_fixture.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using gpt2_test::alternate_index_multiplier;
using gpt2_test::make_checkpoint;
using gpt2_test::make_model_tensors;
using gpt2_test::TemporaryCheckpoint;

using TokenIds = std::vector<std::size_t>;

constexpr std::uint64_t default_seed = 20260905;

int failure_count = 0;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failure_count;
    }
}

void expect_near(
    double actual,
    double expected,
    double tolerance,
    std::string_view message
) {
    if (!std::isfinite(actual) ||
        std::fabs(actual - expected) > tolerance) {
        std::cerr << "FAIL: " << message
                  << " (expected " << expected
                  << " within " << tolerance
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

gpt2::Gpt2Model load_fixture_model() {
    const TemporaryCheckpoint file(
        make_checkpoint(make_model_tensors(alternate_index_multiplier)),
        "gpt2-sampling-test-"
    );
    return gpt2::Gpt2Model(gpt2::load_checkpoint(file.path()));
}

// Scores chosen so that softmax reproduces the probabilities exactly:
// the logarithm of each target probability.
std::vector<float> scores_for(const std::vector<double>& probabilities) {
    std::vector<float> scores;
    scores.reserve(probabilities.size());
    for (const double probability : probabilities) {
        scores.push_back(static_cast<float>(std::log(probability)));
    }
    return scores;
}

const std::vector<double>& reference_probabilities() {
    static const std::vector<double> probabilities{
        0.5, 0.3, 0.15, 0.05
    };
    return probabilities;
}

std::vector<double> draw_frequencies(
    std::span<const float> scores,
    const gpt2::SamplingOptions& options,
    std::size_t draws,
    std::uint64_t seed
) {
    std::mt19937_64 generator(seed);
    std::vector<double> counts(scores.size(), 0.0);
    for (std::size_t draw = 0; draw < draws; ++draw) {
        counts[gpt2::sample_token(scores, options, generator)] += 1.0;
    }

    for (double& count : counts) {
        count /= static_cast<double>(draws);
    }
    return counts;
}

void test_distribution_reproduces_softmax() {
    const std::vector<float> scores =
        scores_for(reference_probabilities());
    const gpt2::SamplingOptions options;
    const std::vector<float> probabilities =
        gpt2::sampling_distribution(scores, options);

    expect(
        probabilities.size() == scores.size(),
        "the distribution covers the whole vocabulary"
    );

    double total = 0.0;
    for (std::size_t token = 0; token < probabilities.size(); ++token) {
        expect_near(
            probabilities[token],
            reference_probabilities()[token],
            1.0e-6,
            "softmax reproduces the intended probability"
        );
        total += probabilities[token];
    }

    expect_near(total, 1.0, 1.0e-6, "the distribution sums to one");
}

void test_temperature_sharpens_and_flattens() {
    const std::vector<float> scores =
        scores_for(reference_probabilities());

    gpt2::SamplingOptions cold;
    cold.temperature = 0.01F;
    const std::vector<float> sharpened =
        gpt2::sampling_distribution(scores, cold);
    expect_near(
        sharpened[0],
        1.0,
        1.0e-6,
        "a low temperature collapses onto the highest score"
    );

    gpt2::SamplingOptions hot;
    hot.temperature = 1000.0F;
    const std::vector<float> flattened =
        gpt2::sampling_distribution(scores, hot);
    for (const float probability : flattened) {
        expect_near(
            probability,
            0.25,
            1.0e-3,
            "a high temperature approaches a uniform distribution"
        );
    }
}

void test_top_k_keeps_ties_at_the_boundary() {
    // Hugging Face keeps every token scoring at least as high as the
    // k-th score, so these three tied scores all survive k = 2.
    const std::array<float, 5> scores{3.0F, 1.0F, 1.0F, 1.0F, 0.0F};

    gpt2::SamplingOptions options;
    options.top_k = 2;
    const std::vector<float> probabilities =
        gpt2::sampling_distribution(scores, options);

    std::size_t kept = 0;
    for (const float probability : probabilities) {
        if (probability > 0.0F) {
            ++kept;
        }
    }

    expect(kept == 4, "a tie at the top-k boundary keeps every tied token");
    expect(
        probabilities[4] == 0.0F,
        "top-k still drops the token below the boundary"
    );
    expect_near(
        probabilities[1],
        probabilities[3],
        1.0e-9,
        "tied tokens keep equal probability"
    );
}

void test_top_k_selects_the_highest_scores() {
    const std::vector<float> scores =
        scores_for(reference_probabilities());

    gpt2::SamplingOptions options;
    options.top_k = 1;
    const std::vector<float> only_best =
        gpt2::sampling_distribution(scores, options);
    expect_near(
        only_best[0],
        1.0,
        1.0e-6,
        "top-k of one keeps the highest-scoring token alone"
    );

    options.top_k = 2;
    const std::vector<float> best_pair =
        gpt2::sampling_distribution(scores, options);
    expect_near(
        best_pair[0],
        0.5 / 0.8,
        1.0e-6,
        "top-k renormalises what it keeps"
    );
    expect_near(
        best_pair[1],
        0.3 / 0.8,
        1.0e-6,
        "top-k renormalises what it keeps"
    );
    expect(
        best_pair[2] == 0.0F && best_pair[3] == 0.0F,
        "top-k drops everything below the boundary"
    );

    options.top_k = 99;
    const std::vector<float> everything =
        gpt2::sampling_distribution(scores, options);
    expect_near(
        everything[3],
        0.05,
        1.0e-6,
        "a top-k beyond the vocabulary changes nothing"
    );
}

void test_top_p_keeps_the_smallest_sufficient_set() {
    const std::vector<float> scores =
        scores_for(reference_probabilities());

    struct Case {
        float top_p;
        std::size_t expected_kept;
        std::string_view message;
    };

    // Cumulative probabilities are 0.5, 0.8, 0.95, 1.0. The thresholds
    // sit between them, away from the boundaries where float rounding
    // would decide the answer.
    const std::array<Case, 4> cases{
        Case{0.4F, 1, "top-p below the first mass keeps one token"},
        Case{0.7F, 2, "top-p inside the second mass keeps two tokens"},
        Case{0.85F, 3, "top-p inside the third mass keeps three tokens"},
        Case{1.0F, 4, "top-p of one keeps the whole vocabulary"},
    };

    for (const Case& test_case : cases) {
        gpt2::SamplingOptions options;
        options.top_p = test_case.top_p;
        const std::vector<float> probabilities =
            gpt2::sampling_distribution(scores, options);

        std::size_t kept = 0;
        for (const float probability : probabilities) {
            if (probability > 0.0F) {
                ++kept;
            }
        }
        expect(kept == test_case.expected_kept, test_case.message);
    }
}

void test_filters_compose() {
    const std::vector<float> scores =
        scores_for(reference_probabilities());

    gpt2::SamplingOptions options;
    options.top_k = 2;
    options.top_p = 0.99F;
    const std::vector<float> probabilities =
        gpt2::sampling_distribution(scores, options);

    std::size_t kept = 0;
    for (const float probability : probabilities) {
        if (probability > 0.0F) {
            ++kept;
        }
    }

    expect(
        kept == 2,
        "top-k applies before top-p, so it bounds the candidate set"
    );
}

void test_draws_match_the_distribution() {
    const std::vector<float> scores =
        scores_for(reference_probabilities());
    constexpr std::size_t draws = 200000;

    const std::vector<double> frequencies = draw_frequencies(
        scores,
        gpt2::SamplingOptions{},
        draws,
        default_seed
    );

    // The standard error at p = 0.5 over 200,000 draws is about 0.0011,
    // so a tolerance of 0.01 is roughly nine standard errors: it will
    // not flake, but it still catches a sampler that ignores weights.
    for (std::size_t token = 0; token < frequencies.size(); ++token) {
        expect_near(
            frequencies[token],
            reference_probabilities()[token],
            0.01,
            "empirical frequencies match the distribution"
        );
    }
}

void test_draws_match_a_renormalised_distribution() {
    const std::vector<float> scores =
        scores_for(reference_probabilities());
    constexpr std::size_t draws = 200000;

    gpt2::SamplingOptions options;
    options.top_p = 0.7F;
    const std::vector<double> frequencies = draw_frequencies(
        scores,
        options,
        draws,
        default_seed
    );

    // Keeping 0.5 and 0.3 and renormalising gives 0.625 and 0.375.
    expect_near(
        frequencies[0],
        0.625,
        0.01,
        "a filtered distribution is renormalised before sampling"
    );
    expect_near(
        frequencies[1],
        0.375,
        0.01,
        "a filtered distribution is renormalised before sampling"
    );
    expect(
        frequencies[2] == 0.0 && frequencies[3] == 0.0,
        "a filtered token is never drawn"
    );
}

void test_sampling_is_reproducible() {
    const gpt2::Gpt2Model model = load_fixture_model();
    const std::array<std::size_t, 1> prompt{1};

    gpt2::GenerationLimits limits;
    limits.maximum_new_tokens = 4;

    gpt2::SamplingOptions options;
    options.temperature = 2.0F;

    std::mt19937_64 first(default_seed);
    std::mt19937_64 second(default_seed);
    std::mt19937_64 third(default_seed + 1);

    const TokenIds from_first = gpt2::generate_sampled(
        model, prompt, limits, options, first
    ).new_token_ids;
    const TokenIds from_second = gpt2::generate_sampled(
        model, prompt, limits, options, second
    ).new_token_ids;

    expect(
        from_first == from_second,
        "the same seed produces the same tokens"
    );
    expect(
        from_first.size() == 4,
        "sampling honours the token limit"
    );

    // Different seeds are only very likely to differ, so this compares
    // many independent draws rather than one sequence.
    const std::vector<double> one = draw_frequencies(
        scores_for(reference_probabilities()),
        gpt2::SamplingOptions{},
        64,
        default_seed
    );
    const std::vector<double> other = draw_frequencies(
        scores_for(reference_probabilities()),
        gpt2::SamplingOptions{},
        64,
        default_seed + 1
    );
    expect(one != other, "a different seed produces different draws");
    static_cast<void>(third);
}

void test_cold_sampling_reproduces_greedy() {
    const gpt2::Gpt2Model model = load_fixture_model();

    gpt2::GenerationLimits limits;
    limits.maximum_new_tokens = 8;

    gpt2::SamplingOptions cold;
    cold.temperature = 0.01F;

    for (const std::size_t first_token : {std::size_t{0}, std::size_t{1},
                                          std::size_t{2}, std::size_t{3},
                                          std::size_t{4}, std::size_t{5},
                                          std::size_t{6}}) {
        const std::array<std::size_t, 1> prompt{first_token};
        std::mt19937_64 generator(default_seed);

        const gpt2::Generation sampled = gpt2::generate_sampled(
            model, prompt, limits, cold, generator
        );
        const gpt2::Generation greedy =
            gpt2::generate_greedy(model, prompt, limits);

        expect(
            sampled.new_token_ids == greedy.new_token_ids,
            "a near-zero temperature reproduces greedy generation"
        );
        expect(
            sampled.stop == greedy.stop,
            "a near-zero temperature stops for the same reason"
        );
    }
}

void test_sampling_stops_at_the_end_of_text_token() {
    const gpt2::Gpt2Model model = load_fixture_model();
    const std::array<std::size_t, 1> prompt{1};

    gpt2::GenerationLimits limits;
    limits.maximum_new_tokens = 8;
    limits.end_of_text_id = 1;

    gpt2::SamplingOptions cold;
    cold.temperature = 0.01F;

    std::mt19937_64 generator(default_seed);
    const gpt2::Generation generation = gpt2::generate_sampled(
        model, prompt, limits, cold, generator
    );

    expect(
        generation.new_token_ids == TokenIds{1},
        "sampling stops on the end-of-text token"
    );
    expect(
        generation.stop == gpt2::GenerationStop::end_of_text,
        "sampling reports the end-of-text stop"
    );
}

void test_sampling_rejects_invalid_options() {
    const std::vector<float> scores =
        scores_for(reference_probabilities());

    const std::array<float, 4> bad_temperatures{
        0.0F,
        -1.0F,
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),
    };
    for (const float temperature : bad_temperatures) {
        gpt2::SamplingOptions options;
        options.temperature = temperature;
        expect_throws<std::invalid_argument>(
            [&scores, options] {
                static_cast<void>(
                    gpt2::sampling_distribution(scores, options)
                );
            },
            "sampling rejects an unusable temperature"
        );
    }

    const std::array<float, 4> bad_top_p{
        0.0F,
        -0.5F,
        1.5F,
        std::numeric_limits<float>::quiet_NaN(),
    };
    for (const float top_p : bad_top_p) {
        gpt2::SamplingOptions options;
        options.top_p = top_p;
        expect_throws<std::invalid_argument>(
            [&scores, options] {
                static_cast<void>(
                    gpt2::sampling_distribution(scores, options)
                );
            },
            "sampling rejects an unusable top_p"
        );
    }

    const std::span<const float> empty;
    expect_throws<std::invalid_argument>(
        [empty] {
            static_cast<void>(
                gpt2::sampling_distribution(empty, gpt2::SamplingOptions{})
            );
        },
        "sampling rejects an empty score vector"
    );

    const std::array<float, 3> all_nan{
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN(),
    };
    expect_throws<std::runtime_error>(
        [&all_nan] {
            static_cast<void>(
                gpt2::sampling_distribution(all_nan, gpt2::SamplingOptions{})
            );
        },
        "sampling rejects a score vector with nothing finite"
    );
}

void test_sampling_ignores_not_a_number_scores() {
    const std::array<float, 4> scores{
        1.0F,
        std::numeric_limits<float>::quiet_NaN(),
        2.0F,
        std::numeric_limits<float>::quiet_NaN(),
    };

    const std::vector<float> probabilities =
        gpt2::sampling_distribution(scores, gpt2::SamplingOptions{});

    expect(
        probabilities[1] == 0.0F && probabilities[3] == 0.0F,
        "a score that is not a number is never sampled"
    );
    expect_near(
        static_cast<double>(probabilities[0]) +
            static_cast<double>(probabilities[2]),
        1.0,
        1.0e-6,
        "the remaining scores carry the whole distribution"
    );
}

}  // namespace

int main() {
    try {
        test_distribution_reproduces_softmax();
        test_temperature_sharpens_and_flattens();
        test_top_k_keeps_ties_at_the_boundary();
        test_top_k_selects_the_highest_scores();
        test_top_p_keeps_the_smallest_sufficient_set();
        test_filters_compose();
        test_draws_match_the_distribution();
        test_draws_match_a_renormalised_distribution();
        test_sampling_is_reproducible();
        test_cold_sampling_reproduces_greedy();
        test_sampling_stops_at_the_end_of_text_token();
        test_sampling_rejects_invalid_options();
        test_sampling_ignores_not_a_number_scores();
    } catch (const std::exception& exception) {
        std::cerr << "FAIL: unexpected exception: "
                  << exception.what() << '\n';
        return EXIT_FAILURE;
    }

    if (failure_count != 0) {
        std::cerr << failure_count
                  << " sampling test assertion(s) failed\n";
        return EXIT_FAILURE;
    }

    std::cout << "All sampling tests passed\n";
    return EXIT_SUCCESS;
}
