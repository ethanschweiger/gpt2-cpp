#include "gpt2/model.h"

#include "model_fixture.h"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
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
using gpt2_test::QuantizationTarget;
using gpt2_test::quantize_model_tensors;
using gpt2_test::QuantizedFixtures;
using gpt2_test::TemporaryCheckpoint;
using gpt2_test::TensorFixture;
using gpt2_test::tied_embedding_target;
using gpt2_test::transformer_weight_targets;

constexpr std::size_t fixture_context_length =
    gpt2_test::fixture_context_length;
constexpr std::size_t fixture_layer_count =
    gpt2_test::fixture_layer_count;

int failure_count = 0;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failure_count;
    }
}

// The cached and uncached paths run the same operations over the same
// values in the same order, so their logits are compared bit for bit
// rather than within a tolerance.
void expect_identical_bits(
    std::span<const float> actual,
    std::span<const float> expected,
    std::string_view message
) {
    if (actual.size() != expected.size()) {
        std::cerr << "FAIL: " << message
                  << " (expected " << expected.size()
                  << " values, got " << actual.size() << ")\n";
        ++failure_count;
        return;
    }

    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (std::bit_cast<std::uint32_t>(actual[index]) !=
            std::bit_cast<std::uint32_t>(expected[index])) {
            std::ostringstream detail;
            detail << message << " (value " << index
                   << ": expected " << expected[index]
                   << ", got " << actual[index] << ")";
            std::cerr << "FAIL: " << detail.str() << '\n';
            ++failure_count;
            return;
        }
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

gpt2::Gpt2Model load_fixture_model(
    std::size_t index_multiplier = alternate_index_multiplier
) {
    const TemporaryCheckpoint file(
        make_checkpoint(make_model_tensors(index_multiplier)),
        "gpt2-kv-cache-test-"
    );
    return gpt2::Gpt2Model(gpt2::load_checkpoint(file.path()));
}

gpt2::Gpt2Model load_model(const std::vector<TensorFixture>& tensors) {
    const TemporaryCheckpoint file(
        make_checkpoint(tensors),
        "gpt2-kv-cache-quantized-test-"
    );
    return gpt2::Gpt2Model(gpt2::load_checkpoint(file.path()));
}

std::span<const float> row_of(
    const gpt2::Tensor& logits,
    std::size_t row
) {
    const std::size_t width = logits.shape()[1];
    return std::span<const float>(logits.data() + row * width, width);
}

std::span<const float> all_of(const gpt2::Tensor& logits) {
    return std::span<const float>(logits.data(), logits.numel());
}

void test_cached_forward_matches_uncached() {
    const gpt2::Gpt2Model model = load_fixture_model();
    const std::array<std::size_t, 4> tokens{2, 5, 1, 3};

    const gpt2::Tensor uncached = model.forward(tokens);

    gpt2::KvCache cache(model.config());
    const gpt2::Tensor cached = model.forward(tokens, cache);

    expect(
        cached.shape() == uncached.shape(),
        "a cached forward pass returns one row per new token"
    );
    expect_identical_bits(
        all_of(cached),
        all_of(uncached),
        "a cached forward pass reproduces the uncached logits exactly"
    );
    expect(
        cache.length() == tokens.size(),
        "the cache holds every token it was given"
    );
}

void test_one_token_at_a_time_matches_uncached() {
    const gpt2::Gpt2Model model = load_fixture_model();
    const std::array<std::size_t, 5> tokens{2, 5, 1, 3, 0};

    const gpt2::Tensor uncached = model.forward(tokens);

    gpt2::KvCache cache(model.config());
    for (std::size_t position = 0; position < tokens.size(); ++position) {
        const std::span<const std::size_t> step(
            tokens.data() + position,
            1
        );
        const gpt2::Tensor cached = model.forward(step, cache);

        expect(
            cached.shape()[0] == 1,
            "a single-token step returns a single row"
        );
        expect_identical_bits(
            all_of(cached),
            row_of(uncached, position),
            "each step reproduces its row of the uncached logits exactly"
        );
        expect(
            cache.length() == position + 1,
            "the cache grows by one token per step"
        );
    }
}

// Mirrors test_one_token_at_a_time_matches_uncached for a fully
// quantized model (transformer weights and wte): quantized_transformer_
// block's cached and uncached overloads share the same append_to_cache/
// extract_cached_head machinery the FP32 block does, so the same
// exact bit-identity should hold at the full-model level, not only at
// the attention/transformer-block level attention_test.cpp and
// transformer_test.cpp already confirm.
void test_quantized_one_token_at_a_time_matches_uncached() {
    std::vector<QuantizationTarget> targets = transformer_weight_targets();
    targets.push_back(tied_embedding_target());
    const QuantizedFixtures fixtures = quantize_model_tensors(targets);
    const gpt2::Gpt2Model model = load_model(fixtures.quantized);

    const std::array<std::size_t, 5> tokens{2, 5, 1, 3, 0};
    const gpt2::Tensor uncached = model.forward(tokens);

    gpt2::KvCache cache(model.config());
    for (std::size_t position = 0; position < tokens.size(); ++position) {
        const std::span<const std::size_t> step(
            tokens.data() + position,
            1
        );
        const gpt2::Tensor cached = model.forward(step, cache);

        expect(
            cached.shape()[0] == 1,
            "a quantized single-token step returns a single row"
        );
        expect_identical_bits(
            all_of(cached),
            row_of(uncached, position),
            "a quantized model's cached step reproduces the uncached "
            "logits exactly"
        );
        expect(
            cache.length() == position + 1,
            "the cache grows by one token per quantized step"
        );
    }
}

void test_mixed_chunk_sizes_match_uncached() {
    const gpt2::Gpt2Model model = load_fixture_model();
    const std::array<std::size_t, 5> tokens{4, 1, 6, 2, 5};

    const gpt2::Tensor uncached = model.forward(tokens);

    // A prompt of two, then one, then two more: the same sequence
    // arriving in different-sized pieces must land in the same place.
    const std::array<std::size_t, 3> chunks{2, 1, 2};
    gpt2::KvCache cache(model.config());
    std::size_t position = 0;

    for (const std::size_t chunk : chunks) {
        const std::span<const std::size_t> step(
            tokens.data() + position,
            chunk
        );
        const gpt2::Tensor cached = model.forward(step, cache);

        for (std::size_t row = 0; row < chunk; ++row) {
            expect_identical_bits(
                row_of(cached, row),
                row_of(uncached, position + row),
                "chunked steps reproduce the uncached logits exactly"
            );
        }

        position += chunk;
    }

    expect(
        cache.length() == tokens.size(),
        "chunked steps fill the cache to the sequence length"
    );
}

// forward_last_token_logits exists purely as an optimization for
// generation: it must return exactly the row forward() would have
// returned for the same final token, never a different one, in both
// the cached and uncached cases and across every sequence length that
// generation actually produces (a multi-token prefill, a single-token
// step, and a sequence already at the context window).
void test_last_token_logits_matches_forwards_last_row() {
    const gpt2::Gpt2Model model = load_fixture_model();
    const std::array<std::size_t, 4> tokens{2, 5, 1, 3};

    const gpt2::Tensor uncached_full = model.forward(tokens);
    const gpt2::Tensor uncached_last =
        model.forward_last_token_logits(tokens);

    expect(
        uncached_last.shape() == gpt2::Tensor::Shape{1, 7},
        "the uncached last-token overload returns a single row"
    );
    expect_identical_bits(
        all_of(uncached_last),
        row_of(uncached_full, tokens.size() - 1),
        "the uncached last-token overload reproduces forward()'s "
        "final row exactly"
    );

    gpt2::KvCache prefill_cache(model.config());
    const gpt2::Tensor cached_full = model.forward(tokens, prefill_cache);

    gpt2::KvCache prefill_cache_again(model.config());
    const gpt2::Tensor cached_last =
        model.forward_last_token_logits(tokens, prefill_cache_again);

    expect(
        cached_last.shape() == gpt2::Tensor::Shape{1, 7},
        "the cached last-token overload returns a single row for a "
        "multi-token prefill"
    );
    expect_identical_bits(
        all_of(cached_last),
        row_of(cached_full, tokens.size() - 1),
        "the cached last-token overload reproduces the prefill's "
        "final row exactly"
    );
    expect(
        prefill_cache_again.length() == prefill_cache.length(),
        "the last-token overload still advances the cache normally"
    );

    // A single-token step is already the "last row" case for the
    // ordinary cached forward(), so the two overloads should agree
    // step for step over a whole generation-style run.
    gpt2::KvCache stepped_cache(model.config());
    gpt2::KvCache stepped_cache_last(model.config());
    for (const std::size_t token : tokens) {
        const std::array<std::size_t, 1> step{token};
        const gpt2::Tensor full_step =
            model.forward(step, stepped_cache);
        const gpt2::Tensor last_step =
            model.forward_last_token_logits(step, stepped_cache_last);

        expect_identical_bits(
            all_of(last_step),
            all_of(full_step),
            "single-token steps agree between both overloads"
        );
    }

    // At the context window's edge there is exactly one row either
    // way, which should not be treated differently from any other
    // sequence length.
    const std::array<std::size_t, 5> full_context{0, 1, 2, 3, 4};
    const gpt2::Tensor edge_full = model.forward(full_context);
    const gpt2::Tensor edge_last =
        model.forward_last_token_logits(full_context);
    expect_identical_bits(
        all_of(edge_last),
        row_of(edge_full, full_context.size() - 1),
        "a sequence at the context window still matches exactly"
    );
}

// A cache used only through forward_last_token_logits must reject and
// roll back exactly as forward() does, since generation relies on
// being able to retry after a rejected call.
void test_last_token_logits_shares_forwards_validation() {
    const gpt2::Gpt2Model model = load_fixture_model();

    const std::span<const std::size_t> empty;
    expect_throws<std::invalid_argument>(
        [&model, empty] {
            static_cast<void>(model.forward_last_token_logits(empty));
        },
        "the uncached last-token overload rejects an empty sequence"
    );

    const std::array<std::size_t, 1> invalid_token{7};
    expect_throws<std::out_of_range>(
        [&model, &invalid_token] {
            static_cast<void>(
                model.forward_last_token_logits(invalid_token)
            );
        },
        "the uncached last-token overload rejects an out-of-range token"
    );

    gpt2::KvCache cache(model.config());
    const std::array<std::size_t, 5> full_context{0, 1, 2, 3, 4};
    static_cast<void>(
        model.forward_last_token_logits(full_context, cache)
    );

    const std::array<std::size_t, 1> one_more{1};
    expect_throws<std::invalid_argument>(
        [&model, &cache, &one_more] {
            static_cast<void>(
                model.forward_last_token_logits(one_more, cache)
            );
        },
        "a full cache still rejects another token through the "
        "last-token overload"
    );
    expect(
        cache.length() == fixture_context_length,
        "a rejected call leaves the cache exactly as full as before"
    );
}

void test_cleared_cache_can_be_reused() {
    const gpt2::Gpt2Model model = load_fixture_model();
    const std::array<std::size_t, 3> first{2, 5, 1};
    const std::array<std::size_t, 3> second{0, 3, 6};

    gpt2::KvCache cache(model.config());
    static_cast<void>(model.forward(first, cache));
    cache.clear();

    expect(cache.length() == 0, "clearing empties the cache");

    const gpt2::Tensor reused = model.forward(second, cache);
    const gpt2::Tensor fresh = model.forward(second);

    expect_identical_bits(
        all_of(reused),
        all_of(fresh),
        "a cleared cache leaves no trace of the earlier sequence"
    );
}

void test_cache_reports_its_shape() {
    const gpt2::Gpt2Model model = load_fixture_model();
    const gpt2::KvCache cache(model.config());

    expect(
        cache.capacity() == fixture_context_length,
        "the cache spans the whole context window"
    );
    expect(
        cache.layer_count() == fixture_layer_count,
        "the cache holds one entry per transformer layer"
    );
    expect(cache.length() == 0, "a new cache is empty");
}

void test_populated_cache_is_bound_to_its_model() {
    const gpt2::Gpt2Model first = load_fixture_model();
    const gpt2::Gpt2Model second = load_fixture_model(
        gpt2_test::standard_index_multiplier
    );
    const std::array<std::size_t, 2> tokens{2, 5};

    gpt2::KvCache cache(first.config());
    static_cast<void>(first.forward(tokens, cache));

    expect_throws<std::invalid_argument>(
        [&second, &cache, &tokens] {
            static_cast<void>(second.forward(tokens, cache));
        },
        "a populated cache cannot be mixed with another model"
    );

    cache.clear();
    const gpt2::Tensor reused = second.forward(tokens, cache);
    const gpt2::Tensor fresh = second.forward(tokens);
    expect_identical_bits(
        all_of(reused),
        all_of(fresh),
        "clearing lets a cache bind to another compatible model"
    );
}

void test_cache_rejects_invalid_use() {
    const gpt2::Gpt2Model model = load_fixture_model();

    gpt2::KvCache cache(model.config());
    const std::array<std::size_t, 5> full_context{0, 1, 2, 3, 4};
    static_cast<void>(model.forward(full_context, cache));

    expect(
        cache.length() == fixture_context_length,
        "the cache fills to the context window"
    );

    const std::array<std::size_t, 1> one_more{1};
    expect_throws<std::invalid_argument>(
        [&model, &cache, &one_more] {
            static_cast<void>(model.forward(one_more, cache));
        },
        "a full cache rejects another token"
    );
    expect(
        cache.length() == fixture_context_length,
        "a rejected call leaves the cache exactly as full as before"
    );

    gpt2::KvCache empty_cache(model.config());
    const std::span<const std::size_t> empty;
    expect_throws<std::invalid_argument>(
        [&model, &empty_cache, empty] {
            static_cast<void>(model.forward(empty, empty_cache));
        },
        "a cached forward pass rejects an empty token sequence"
    );

    const std::array<std::size_t, 1> invalid_token{7};
    expect_throws<std::out_of_range>(
        [&model, &empty_cache, &invalid_token] {
            static_cast<void>(model.forward(invalid_token, empty_cache));
        },
        "a cached forward pass rejects a token outside the vocabulary"
    );

    gpt2::ModelConfig other = model.config();
    other.layer_count += 1;
    gpt2::KvCache mismatched(other);
    const std::array<std::size_t, 1> token{1};
    expect_throws<std::invalid_argument>(
        [&model, &mismatched, &token] {
            static_cast<void>(model.forward(token, mismatched));
        },
        "a cache with an incompatible layer count is rejected"
    );

    gpt2::ModelConfig narrower = model.config();
    narrower.embedding_size = 2;
    narrower.head_count = 1;
    gpt2::KvCache narrow(narrower);
    expect_throws<std::invalid_argument>(
        [&model, &narrow, &token] {
            static_cast<void>(model.forward(token, narrow));
        },
        "a cache with the wrong width is rejected"
    );
}

}  // namespace

int main() {
    try {
        test_cached_forward_matches_uncached();
        test_one_token_at_a_time_matches_uncached();
        test_quantized_one_token_at_a_time_matches_uncached();
        test_mixed_chunk_sizes_match_uncached();
        test_last_token_logits_matches_forwards_last_row();
        test_last_token_logits_shares_forwards_validation();
        test_cleared_cache_can_be_reused();
        test_cache_reports_its_shape();
        test_populated_cache_is_bound_to_its_model();
        test_cache_rejects_invalid_use();
    } catch (const std::exception& exception) {
        std::cerr << "FAIL: unexpected exception: "
                  << exception.what() << '\n';
        return EXIT_FAILURE;
    }

    if (failure_count != 0) {
        std::cerr << failure_count
                  << " key/value cache test assertion(s) failed\n";
        return EXIT_FAILURE;
    }

    std::cout << "All key/value cache tests passed\n";
    return EXIT_SUCCESS;
}
