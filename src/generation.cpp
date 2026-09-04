#include "gpt2/generation.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace gpt2 {

namespace {

std::size_t most_likely_next_token(const Tensor& logits) {
    const Tensor::Shape& shape = logits.shape();
    if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
        throw std::logic_error(
            "model returned logits with an unexpected shape"
        );
    }

    const std::size_t vocabulary_size = shape[1];
    const std::size_t last_row = (shape[0] - 1) * vocabulary_size;

    // A strict comparison keeps the lowest ID when scores tie, and
    // skips any NaN score because every comparison against one fails.
    std::size_t best_token = 0;
    float best_score = logits.at(last_row);
    for (std::size_t token = 1; token < vocabulary_size; ++token) {
        const float score = logits.at(last_row + token);
        if (score > best_score) {
            best_score = score;
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

}  // namespace

GreedyGeneration generate_greedy(
    const Gpt2Model& model,
    std::span<const std::size_t> prompt_token_ids,
    const GreedyGenerationOptions& options
) {
    if (prompt_token_ids.empty()) {
        throw std::invalid_argument(
            "greedy generation requires at least one prompt token"
        );
    }

    const std::size_t context_length =
        static_cast<std::size_t>(model.config().context_length);
    if (prompt_token_ids.size() > context_length) {
        throw std::invalid_argument(
            "prompt exceeds the checkpoint context length"
        );
    }

    std::vector<std::size_t> sequence;
    sequence.reserve(context_length);
    sequence.assign(prompt_token_ids.begin(), prompt_token_ids.end());

    GreedyGeneration generation;
    generation.new_token_ids.reserve(std::min(
        options.maximum_new_tokens,
        context_length - prompt_token_ids.size()
    ));

    while (generation.new_token_ids.size() < options.maximum_new_tokens) {
        if (sequence.size() >= context_length) {
            generation.stop = GenerationStop::context_limit;
            return generation;
        }

        const std::size_t next_token =
            most_likely_next_token(model.forward(sequence));
        sequence.push_back(next_token);
        generation.new_token_ids.push_back(next_token);

        if (options.end_of_text_id.has_value() &&
            next_token == *options.end_of_text_id) {
            generation.stop = GenerationStop::end_of_text;
            return generation;
        }
    }

    generation.stop = GenerationStop::token_limit;
    return generation;
}

}  // namespace gpt2
