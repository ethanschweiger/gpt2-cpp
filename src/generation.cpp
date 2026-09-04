#include "gpt2/generation.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace gpt2 {

namespace {

constexpr double negative_infinity =
    -std::numeric_limits<double>::infinity();

// Turns one draw from the engine into a double in [0, 1). Doing this by
// hand rather than with std::uniform_real_distribution keeps a seeded
// run reproducible, because the standard fixes the engine's output but
// not any distribution's algorithm.
double next_unit_double(std::mt19937_64& generator) {
    constexpr double scale = 1.0 / 9007199254740992.0;  // 2^-53
    return static_cast<double>(generator() >> 11U) * scale;
}

std::span<const float> last_row(const Tensor& logits) {
    const Tensor::Shape& shape = logits.shape();
    if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
        throw std::logic_error(
            "model returned logits with an unexpected shape"
        );
    }

    const std::size_t vocabulary_size = shape[1];
    return std::span<const float>(
        logits.data() + (shape[0] - 1) * vocabulary_size,
        vocabulary_size
    );
}

std::size_t most_likely_token(std::span<const float> scores) {
    // A strict comparison keeps the lowest ID when scores tie, and
    // skips any NaN score because every comparison against one fails.
    std::size_t best_token = 0;
    float best_score = scores[0];
    for (std::size_t token = 1; token < scores.size(); ++token) {
        if (scores[token] > best_score) {
            best_score = scores[token];
            best_token = token;
        }
    }

    if (!std::isfinite(best_score)) {
        throw std::runtime_error(
            "model produced no finite logit to generate from"
        );
    }

    return best_token;
}

void validate_sampling_options(const SamplingOptions& options) {
    if (!std::isfinite(options.temperature) ||
        options.temperature <= 0.0F) {
        throw std::invalid_argument(
            "sampling temperature must be finite and greater than zero"
        );
    }

    if (!std::isfinite(options.top_p) ||
        options.top_p <= 0.0F ||
        options.top_p > 1.0F) {
        throw std::invalid_argument(
            "sampling top_p must lie between zero and one"
        );
    }
}

// Softmax over the scores, skipping the tokens already ruled out.
// Subtracting the highest score first keeps the exponentials in range,
// and a ruled-out score of negative infinity becomes zero.
//
// The work happens in double even though the scores arrive as float,
// because a sharpening temperature spreads them far enough that a float
// subtraction would cost several digits in the exponent.
std::vector<double> softmax(std::span<const double> scores) {
    double highest = negative_infinity;
    for (const double score : scores) {
        if (score > highest) {
            highest = score;
        }
    }

    std::vector<double> probabilities(scores.size(), 0.0);
    double total = 0.0;
    for (std::size_t token = 0; token < scores.size(); ++token) {
        if (scores[token] == negative_infinity) {
            continue;
        }

        const double weight = std::exp(scores[token] - highest);
        probabilities[token] = weight;
        total += weight;
    }

    if (!(total > 0.0)) {
        throw std::runtime_error(
            "no token kept a usable probability"
        );
    }

    for (double& probability : probabilities) {
        probability /= total;
    }

    return probabilities;
}

// Keeps every token scoring at least as high as the top_k-th score.
// Hugging Face compares against that score rather than truncating a
// sorted list, so a tie at the boundary keeps more than top_k tokens.
void apply_top_k(std::vector<double>& scores, std::size_t top_k) {
    if (top_k == 0 || top_k >= scores.size()) {
        return;
    }

    std::vector<double> ordered(scores);
    std::nth_element(
        ordered.begin(),
        ordered.begin() + static_cast<std::ptrdiff_t>(top_k - 1),
        ordered.end(),
        std::greater<double>()
    );
    const double threshold = ordered[top_k - 1];

    for (double& score : scores) {
        if (score < threshold) {
            score = negative_infinity;
        }
    }
}

// Drops the least likely tokens while their running probability stays
// at or below 1 - top_p, always keeping the highest-scoring one.
// Accumulating from the least likely token is what Hugging Face does;
// accumulating from the most likely one is equivalent in exact
// arithmetic but disagrees in floating point at the boundary.
void apply_top_p(std::vector<double>& scores, float top_p) {
    if (top_p >= 1.0F) {
        return;
    }

    const std::vector<double> probabilities = softmax(scores);

    std::vector<std::size_t> ascending(scores.size());
    std::iota(ascending.begin(), ascending.end(), std::size_t{0});
    std::sort(
        ascending.begin(),
        ascending.end(),
        [&scores](std::size_t left, std::size_t right) {
            if (scores[left] != scores[right]) {
                return scores[left] < scores[right];
            }
            return left < right;
        }
    );

    const double limit = 1.0 - static_cast<double>(top_p);
    double cumulative = 0.0;
    for (std::size_t position = 0;
         position + 1 < ascending.size();
         ++position) {
        const std::size_t token = ascending[position];
        cumulative += probabilities[token];
        if (cumulative <= limit) {
            scores[token] = negative_infinity;
        }
    }
}

}  // namespace

std::vector<float> sampling_distribution(
    std::span<const float> scores,
    const SamplingOptions& options
) {
    if (scores.empty()) {
        throw std::invalid_argument(
            "sampling requires at least one score"
        );
    }
    validate_sampling_options(options);

    std::vector<double> working(scores.size());
    const double temperature = static_cast<double>(options.temperature);
    bool any_finite = false;
    for (std::size_t token = 0; token < scores.size(); ++token) {
        if (std::isnan(scores[token])) {
            working[token] = negative_infinity;
            continue;
        }

        working[token] = static_cast<double>(scores[token]) / temperature;
        any_finite = any_finite || std::isfinite(working[token]);
    }

    if (!any_finite) {
        throw std::runtime_error(
            "no finite score is available to sample from"
        );
    }

    apply_top_k(working, options.top_k);
    apply_top_p(working, options.top_p);

    const std::vector<double> probabilities = softmax(working);
    std::vector<float> result(probabilities.size());
    for (std::size_t token = 0; token < probabilities.size(); ++token) {
        result[token] = static_cast<float>(probabilities[token]);
    }

    return result;
}

std::size_t sample_token(
    std::span<const float> scores,
    const SamplingOptions& options,
    std::mt19937_64& generator
) {
    const std::vector<float> probabilities =
        sampling_distribution(scores, options);

    const double target = next_unit_double(generator);
    double cumulative = 0.0;
    for (std::size_t token = 0; token < probabilities.size(); ++token) {
        cumulative += static_cast<double>(probabilities[token]);
        if (target < cumulative) {
            return token;
        }
    }

    // Rounding can leave the total a hair below the draw, so fall back
    // to the last token that could have been chosen.
    for (std::size_t token = probabilities.size(); token > 0; --token) {
        if (probabilities[token - 1] > 0.0F) {
            return token - 1;
        }
    }

    throw std::runtime_error("no token kept a usable probability");
}

namespace {

template <typename ChooseToken>
Generation run_generation(
    const Gpt2Model& model,
    std::span<const std::size_t> prompt_token_ids,
    const GenerationLimits& limits,
    ChooseToken choose_token
) {
    if (prompt_token_ids.empty()) {
        throw std::invalid_argument(
            "generation requires at least one prompt token"
        );
    }

    const std::size_t context_length =
        static_cast<std::size_t>(model.config().context_length);
    if (prompt_token_ids.size() > context_length) {
        throw std::invalid_argument(
            "prompt exceeds the checkpoint context length"
        );
    }

    Generation generation;
    generation.new_token_ids.reserve(std::min(
        limits.maximum_new_tokens,
        context_length - prompt_token_ids.size()
    ));

    // The cached path feeds the prompt once and then one token per
    // step; the uncached path replays the whole sequence every time.
    std::optional<KvCache> cache;
    std::vector<std::size_t> sequence;
    std::vector<std::size_t> pending;

    if (limits.use_cache) {
        cache.emplace(model.config());
        pending.assign(
            prompt_token_ids.begin(),
            prompt_token_ids.end()
        );
    } else {
        sequence.reserve(context_length);
        sequence.assign(
            prompt_token_ids.begin(),
            prompt_token_ids.end()
        );
    }

    std::size_t length = prompt_token_ids.size();

    while (generation.new_token_ids.size() < limits.maximum_new_tokens) {
        if (length >= context_length) {
            generation.stop = GenerationStop::context_limit;
            return generation;
        }

        const Tensor logits = cache.has_value()
            ? model.forward(pending, *cache)
            : model.forward(sequence);
        const std::size_t next_token = choose_token(last_row(logits));

        if (cache.has_value()) {
            pending.assign(1, next_token);
        } else {
            sequence.push_back(next_token);
        }
        ++length;
        generation.new_token_ids.push_back(next_token);

        if (limits.end_of_text_id.has_value() &&
            next_token == *limits.end_of_text_id) {
            generation.stop = GenerationStop::end_of_text;
            return generation;
        }
    }

    generation.stop = GenerationStop::token_limit;
    return generation;
}

}  // namespace

Generation generate_greedy(
    const Gpt2Model& model,
    std::span<const std::size_t> prompt_token_ids,
    const GenerationLimits& limits
) {
    return run_generation(
        model,
        prompt_token_ids,
        limits,
        [](std::span<const float> scores) {
            return most_likely_token(scores);
        }
    );
}

Generation generate_sampled(
    const Gpt2Model& model,
    std::span<const std::size_t> prompt_token_ids,
    const GenerationLimits& limits,
    const SamplingOptions& sampling,
    std::mt19937_64& generator
) {
    validate_sampling_options(sampling);

    return run_generation(
        model,
        prompt_token_ids,
        limits,
        [&sampling, &generator](std::span<const float> scores) {
            return sample_token(scores, sampling, generator);
        }
    );
}

}  // namespace gpt2
