#pragma once

#include "gpt2/attention.h"
#include "gpt2/checkpoint.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace gpt2 {

// Keys and values for every layer, so that a run can extend a sequence
// without recomputing the tokens it has already seen.
class KvCache {
public:
    explicit KvCache(const ModelConfig& config);

    // Tokens already stored. Every layer holds the same number.
    std::size_t length() const;
    std::size_t capacity() const;
    std::size_t layer_count() const;

    void clear();

private:
    friend class Gpt2Model;

    std::vector<AttentionCache> layers_m;
    std::uint32_t embedding_size_m;
    std::uint32_t head_count_m;
    std::uint64_t owner_id_m = 0;
};

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

    // Appends the tokens to the cache and returns the logits for those
    // tokens alone, so a run of length n costs n single-token steps
    // rather than n growing forward passes. A populated cache is bound
    // to the logical model that filled it until clear() is called.
    Tensor forward(
        std::span<const std::size_t> token_ids,
        KvCache& cache
    ) const;

private:
    Checkpoint checkpoint_m;
    std::uint64_t identity_m;
};

}  // namespace gpt2
