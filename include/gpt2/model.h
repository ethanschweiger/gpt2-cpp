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

    // Returns [1, vocabulary size] logits for only the final position
    // of token_ids. Generation never reads any other row of forward's
    // result (see gpt2::last_row in generation.cpp), so projecting
    // every earlier position onto the vocabulary is pure waste there;
    // see docs/profiling.md for how much. Prefer this over forward()
    // whenever only the next-token distribution is needed. The last
    // row it returns is bit-identical to forward()'s.
    Tensor forward_last_token_logits(
        std::span<const std::size_t> token_ids
    ) const;

    Tensor forward_last_token_logits(
        std::span<const std::size_t> token_ids,
        KvCache& cache
    ) const;

private:
    Tensor run_transformer_stack(
        std::span<const std::size_t> token_ids
    ) const;

    // Checks that the cache matches this model and has room for
    // token_ids, and returns the cache's current length. Throws before
    // any mutation, so callers call this before their own try/catch —
    // a call that is rejected here never needs a rollback.
    std::size_t validate_cached_forward(
        std::span<const std::size_t> token_ids,
        const KvCache& cache
    ) const;

    // Runs every transformer layer, mutating cache in place, and
    // returns the final (post layer_norm) hidden state for token_ids.
    // The caller must have already validated the call and computed
    // sequence_start via validate_cached_forward; if this throws
    // partway through, cache.layers_m may hold a mix of updated and
    // un-updated lengths, and the caller is expected to roll every
    // layer back to sequence_start.
    Tensor run_cached_transformer_stack(
        std::span<const std::size_t> token_ids,
        KvCache& cache,
        std::size_t sequence_start
    ) const;

    Checkpoint checkpoint_m;
    std::uint64_t identity_m;
};

}  // namespace gpt2
