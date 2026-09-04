# Generation

`gpt2::generate_greedy` extends a prompt one token at a time, always
taking the highest-scoring token from the model's last position.

```cpp
#include "gpt2/generation.h"

gpt2::GenerationLimits limits;
limits.maximum_new_tokens = 20;
limits.end_of_text_id = tokenizer.end_of_text_id();

const gpt2::Generation generation = gpt2::generate_greedy(
    model,
    tokenizer.encode("The capital of France is"),
    limits
);

const std::string continuation =
    tokenizer.decode(generation.new_token_ids);
```

`generate_greedy` returns only the appended tokens, so the caller keeps
the prompt it already has. `generate_sampled` is the same loop with a
draw in place of the argmax; see [Sampling](#sampling).

For a complete checkpoint-to-text command, see
[Command-line generation](cli.md).

## How a step works

The default cached path runs the prompt once, then sends one new token
through the model at each step while reusing the earlier keys and values.
The uncached comparison path runs the full forward pass over the growing
sequence. Both read the last row of the
`[sequence length, vocabulary size]` logits and take its highest-scoring
index.

Ties break toward the lower token ID. Non-finite scores are skipped; if
no finite score remains, the call throws `std::runtime_error` rather than
returning an arbitrary token.

## Stopping

Generation stops for one of three reasons, reported in
`Generation::stop`:

| `GenerationStop` | Meaning |
| --- | --- |
| `token_limit` | `maximum_new_tokens` tokens were generated |
| `end_of_text` | the configured end-of-text token was chosen, and is included in the result |
| `context_limit` | the sequence reached the checkpoint's context window |

`end_of_text_id` is optional. Leave it unset to run to one of the other
two limits. `Gpt2Tokenizer::end_of_text_id()` supplies the right value
for GPT-2, which is 50256.

An empty prompt and a prompt longer than the context window are both
rejected with `std::invalid_argument`; a prompt that already fills the
context window is not an error and simply generates nothing.

## Sampling

`generate_sampled` replaces the argmax with a draw from a filtered
distribution.

```cpp
gpt2::SamplingOptions sampling;
sampling.temperature = 0.8F;
sampling.top_k = 40;
sampling.top_p = 0.95F;

std::mt19937_64 generator(20260905);
const gpt2::Generation generation = gpt2::generate_sampled(
    model, prompt_token_ids, limits, sampling, generator
);
```

The generator supplies every random draw, so seeding it fixes the whole
run. Two lower-level entry points are public because they are worth
testing on their own: `sampling_distribution` applies the filters and
returns the probability of every token, and `sample_token` draws from
that distribution.

Each step applies temperature, then top-k, then top-p, in that order.

**Temperature** divides the scores. Values below one sharpen the
distribution toward the highest score; values above one flatten it. It
must be finite and greater than zero — there is no "temperature of zero
means greedy" special case, because that would make the deterministic
path depend on a floating-point comparison against zero. Use
`generate_greedy` instead, or a temperature small enough to collapse the
distribution; the unit tests check that 0.01 reproduces greedy output
exactly.

**Top-k** keeps every token scoring at least as high as the k-th score.
That is not the same as keeping k tokens: scores of `[3, 1, 1, 1, 0]`
with `top_k = 2` keep **four** tokens, because three of them tie at the
boundary. This matches the reference implementation, which compares
against the k-th score rather than truncating a sorted list.

**Top-p** keeps the most likely tokens whose probabilities reach p, and
always keeps at least one. It accumulates from the *least* likely token
and drops while the running total stays at or below `1 - p`. Walking
down from the most likely token instead is equivalent in exact
arithmetic and disagrees in floating point: measured against the
reference over 12,800 configurations, the descending rule differed on
218 of them and the ascending rule on none.

Scores that are not a number are dropped rather than sampled, and a
score vector with nothing finite in it throws instead of returning an
arbitrary token.

### Reproducibility

The uniform draw is taken from the engine directly rather than through
`std::uniform_real_distribution`, because the C++ standard fixes the
output of `std::mt19937_64` but not the algorithm any distribution uses
to consume it. This makes the mapping from engine output to uniform draw
portable across standard libraries. The full sampled sequence can still
vary across platforms if math-library rounding moves a cumulative
probability across a draw boundary.

### Precision

The filters work in double even though the scores arrive as float. A
sharpening temperature spreads the scores far enough that a float
subtraction inside the softmax costs several digits in the exponent, and
summing 50,257 terms in float loses more. The result is rounded back to
float only at the end.

That makes GPT2CPP *more* accurate than a float32 reference, which is
worth stating precisely, because it means an exact comparison against
one would be measuring the reference's rounding:

| comparison | largest probability difference |
| --- | --- |
| GPT2CPP versus a float64 reference | 3.45e-08 |
| GPT2CPP versus a float32 reference | 8.26e-06 |
| float32 versus float64 reference | 8.26e-06 |

The second and third rows agree to two digits, which says the whole gap
between GPT2CPP and the float32 reference is that reference's own
rounding error. Measured against the accurate answer, GPT2CPP is about
240 times closer.

## The key/value cache

Attention over a sequence recomputes the same keys and values every
step: appending one token does not change what the earlier ones project
to. `GenerationLimits::use_cache` (on by default) keeps them.

```cpp
gpt2::KvCache cache(model.config());
const gpt2::Tensor first = model.forward(prompt_token_ids, cache);
const gpt2::Tensor next = model.forward(one_token, cache);
```

`Gpt2Model::forward` has an overload that appends tokens to a cache and
returns the logits for those tokens alone. A cache is sized for the
whole context window at construction and validated against the model's
configuration on every call. Its first successful forward pass also
binds it to that logical model, so populated state cannot accidentally
be mixed with another model's weights. Calling `clear()` removes both the
sequence and that binding.

Without the cache, generating `n` tokens from a prompt of `p` repeats
full forward passes over `p`, `p + 1`, … tokens; attention work across
those growing passes is cubic in sequence length. With the cache, the
prompt is processed once and each new token takes a single-token step.
That step still attends over all earlier keys, so its attention work
grows linearly with the cached context, but it no longer reprojects or
runs the MLP for every earlier token.

### Cost

Measured on GPT-2 Small, Release build, 8-token prompt extended by 24
tokens:

| | total | per token |
| --- | --- | --- |
| without cache | 50.16 s | 2.090 s |
| with cache | 4.95 s | 0.206 s |

This preliminary run was **10.1× faster**, generating identical tokens.
It is a one-run functional benchmark, not yet a stable resume metric:
timings depend on the machine and run conditions, and the profiling phase
will add warm-ups, repeated trials, summary statistics and environment
metadata.

`tests/kv_cache_benchmark.cpp` performs this comparison and fails if the
two paths disagree. A rerun reports timings for the current machine and
conditions rather than expecting these exact values.

### Why the two paths agree exactly

They do not merely agree closely — the logits are **bit for bit
identical**, and `tests/kv_cache_test.cpp` asserts that on raw float
bits rather than within a tolerance. Three properties make that true:

- `matmul` accumulates strictly sequentially over the inner dimension,
  and that dimension is the head size in both paths, so every dot
  product is the same sum in the same order.
- In the uncached path the causal mask sets future scores to negative
  infinity, and `softmax` turns those into exact zeros. Adding zeros to
  the running total leaves a float sum unchanged, so the denominator
  matches the cached path's shorter sum exactly.
- Multiplying those zero weights by the corresponding values
  contributes exact zeros to the output as well.

An implementation that reordered any of those sums would still be
*correct*, but it would no longer be bit-identical, and the test would
say so.

## Testing

Unit tests run against a small fixture model built in memory by
`tests/model_fixture.h`. Its weights are a deterministic function of
their position, so the same model can be rebuilt in PyTorch, and every
expectation in `tests/generation_test.cpp` comes from running
`GPT2LMHeadModel.generate(do_sample=False)` on that rebuilt model.

The fixture is parameterized so that the generation tests use a variant
whose greedy continuation *changes* between steps. That matters: with a
model that repeats one token forever, an implementation that reused the
first step's logits would still pass.

### Sampling parity

A second opt-in test compares the filters themselves against Hugging
Face's `TemperatureLogitsWarper`, `TopKLogitsWarper` and
`TopPLogitsWarper` across vocabulary sizes from 2 to 50,257, six
temperatures, six top-k values and seven top-p values, plus cases with
deliberately tied scores.

```bash
cmake -S . -B build-sampling \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPT2_WARNINGS_AS_ERRORS=ON \
  -DGPT2_ENABLE_SAMPLING_PARITY=ON \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
```

```bash
cmake --build build-sampling --parallel
ctest --test-dir build-sampling -L sampling --output-on-failure -V
```

Which tokens survive is compared exactly; the probabilities are compared
against a float64 reference for the reason given above. Two cases are
classified rather than failed, because the reference does not define
them:

- A token whose probability is smaller than float32 can represent reads
  as zero once stored. That is underflow, not a filter disagreement.
- When scores tie at the top-p boundary, which of the equally likely
  tokens survives depends on the sort order, and `torch.sort` is not
  stable. The kept count and the kept distribution are still well
  defined, so those are compared instead.

#### Baseline result

```text
filter configurations compared:                          1,828
probabilities compared:                             13,638,216
kept-token-set mismatches:                                   0
tokens too small to store as float32:                  125,263
configurations where tied scores make the boundary ambiguous: 45
largest probability difference from the float64 reference: 3.45e-08
mismatches:                                                  0
```

### End-to-end parity

The opt-in parity test drives the whole pipeline — text in, tokenizer,
model, greedy loop, tokenizer out — and compares it with the same
pipeline in Hugging Face. Every prompt runs twice, once with the cache
and once without, so both paths are checked against the reference and
against each other.

Reproducibility controls match [Numerical Validation](numerical-validation.md)
and [Tokenizer](tokenizer.md): the pinned revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`, checksummed assets, CPU FP32
inference in evaluation mode, eager attention, one PyTorch compute
thread, deterministic algorithms and offline loading.

```bash
cmake -S . -B build-generation \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPT2_WARNINGS_AS_ERRORS=ON \
  -DGPT2_ENABLE_GENERATION_PARITY=ON \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
```

```bash
cmake --build build-generation --parallel
ctest --test-dir build-generation -L generation --output-on-failure -V
```

The test compares generated token IDs exactly, not decoded text, so a
single divergent token fails it and the report names the position where
the two runs first differ.

### Baseline result

```text
GPT2CPP context length:          1,024
cases:                               3
paths compared per case:             2
generated tokens compared:          32
mismatches:                          0
cached-versus-uncached mismatches:   0
```

```text
'Hello, world!'            -> "Hello, world!\n\nI'm sorry, but I'm not sure what"
'The capital of France is' -> 'The capital of France is the capital of the French Republic, and the capital of the'
'1, 2, 3, 4,'              -> '1, 2, 3, 4, 5, 6, 7, 8,'
```

Every one of the 32 generated tokens matches Hugging Face. Unlike the
forward pass, this comparison has no tolerance: greedy decoding turns
the model's small floating-point differences into a discrete choice, so
a large enough error would flip a token and fail the test outright.
