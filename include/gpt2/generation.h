#pragma once

#include "gpt2/model.h"

#include <cstddef>
#include <optional>
#include <random>
#include <span>
#include <vector>

namespace gpt2 {

// Why generation stopped.
enum class GenerationStop {
    token_limit,
    end_of_text,
    context_limit,
};

struct GenerationLimits {
    std::size_t maximum_new_tokens = 0;

    // Generation stops as soon as this token is chosen, and the token
    // is part of the result. Leave it empty to run to another limit.
    std::optional<std::size_t> end_of_text_id;

    // Reuse the keys and values of earlier tokens instead of running
    // the whole sequence again at every step. The two paths produce
    // identical tokens; the uncached one exists for differential
    // testing and for measuring what the cache is worth.
    bool use_cache = true;
};

struct Generation {
    std::vector<std::size_t> new_token_ids;
    GenerationStop stop = GenerationStop::token_limit;
};

struct SamplingOptions {
    // Divides the scores before they become probabilities. Lower values
    // sharpen the distribution toward the highest-scoring token.
    float temperature = 1.0F;

    // Keeps every token scoring at least as high as the top_k-th score,
    // so a tie at that score keeps more than top_k tokens. Zero keeps
    // the whole vocabulary.
    std::size_t top_k = 0;

    // Keeps the most likely tokens whose probabilities reach top_p, and
    // always keeps at least one. A value of one keeps everything.
    float top_p = 1.0F;
};

// Extends the prompt one token at a time, always taking the
// highest-scoring token and breaking ties toward the lower ID. Returns
// only the appended tokens.
//
// By default the prompt is processed once and later steps reuse its
// cached keys and values. Set GenerationLimits::use_cache to false to
// replay the growing sequence at every step.
Generation generate_greedy(
    const Gpt2Model& model,
    std::span<const std::size_t> prompt_token_ids,
    const GenerationLimits& limits
);

// The same loop, drawing each token from the filtered distribution
// instead of taking the highest score. The generator supplies every
// random draw, so seeding it fixes the whole run.
Generation generate_sampled(
    const Gpt2Model& model,
    std::span<const std::size_t> prompt_token_ids,
    const GenerationLimits& limits,
    const SamplingOptions& sampling,
    std::mt19937_64& generator
);

// Applies temperature, top-k and top-p to one row of scores and returns
// the probability of every token in the vocabulary. Filtered-out tokens
// have probability zero and the result sums to one.
std::vector<float> sampling_distribution(
    std::span<const float> scores,
    const SamplingOptions& options
);

// Draws one token from that distribution.
std::size_t sample_token(
    std::span<const float> scores,
    const SamplingOptions& options,
    std::mt19937_64& generator
);

}  // namespace gpt2
