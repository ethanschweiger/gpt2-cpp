#pragma once

#include "gpt2/model.h"

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace gpt2 {

// Why generation stopped.
enum class GenerationStop {
    token_limit,
    end_of_text,
    context_limit,
};

struct GreedyGenerationOptions {
    std::size_t maximum_new_tokens = 0;

    // Generation stops as soon as this token is chosen, and the token
    // is part of the result. Leave it empty to run to another limit.
    std::optional<std::size_t> end_of_text_id;
};

struct GreedyGeneration {
    std::vector<std::size_t> new_token_ids;
    GenerationStop stop = GenerationStop::token_limit;
};

// Extends the prompt one token at a time, always taking the
// highest-scoring token and breaking ties toward the lower ID. Returns
// only the appended tokens.
//
// Every step re-runs the whole forward pass, so a run costs one forward
// pass per new token over a sequence that grows by one each time.
GreedyGeneration generate_greedy(
    const Gpt2Model& model,
    std::span<const std::size_t> prompt_token_ids,
    const GreedyGenerationOptions& options
);

}  // namespace gpt2
