#pragma once

#include "gpt2/checkpoint.h"

#include <cstddef>
#include <span>

namespace gpt2 {

class Gpt2Model {
public:
    explicit Gpt2Model(Checkpoint checkpoint);

    Gpt2Model(const Gpt2Model&) = delete;
    Gpt2Model& operator=(const Gpt2Model&) = delete;
    Gpt2Model(Gpt2Model&&) = default;
    Gpt2Model& operator=(Gpt2Model&&) = default;

    const ModelConfig& config() const;

    // Returns [sequence length, vocabulary size] logits.
    Tensor forward(std::span<const std::size_t> token_ids) const;

private:
    Checkpoint checkpoint_m;
};

}  // namespace gpt2
