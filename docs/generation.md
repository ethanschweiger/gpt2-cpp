# Generation

`gpt2::generate_greedy` extends a prompt one token at a time, always
taking the highest-scoring token from the model's last position.

```cpp
#include "gpt2/generation.h"

gpt2::GreedyGenerationOptions options;
options.maximum_new_tokens = 20;
options.end_of_text_id = tokenizer.end_of_text_id();

const gpt2::GreedyGeneration generation = gpt2::generate_greedy(
    model,
    tokenizer.encode("The capital of France is"),
    options
);

const std::string continuation =
    tokenizer.decode(generation.new_token_ids);
```

`generate_greedy` returns only the appended tokens, so the caller keeps
the prompt it already has.

## How a step works

Each step runs the full forward pass over the whole sequence so far,
reads the last row of the `[sequence length, vocabulary size]` logits,
and takes its highest-scoring index. That token is appended to the
sequence and the next step runs again over the longer sequence.

Ties break toward the lower token ID. `NaN` scores are skipped, because
every comparison against one is false; if no finite score remains, the
call throws `std::runtime_error` rather than returning an arbitrary
token.

## Stopping

Generation stops for one of three reasons, reported in
`GreedyGeneration::stop`:

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

## Cost

There is no key/value cache yet, so the work is quadratic in the
sequence length: generating `n` tokens from a prompt of `p` costs one
full forward pass over `p`, then `p + 1`, and so on. On this machine a
Release build spends roughly 0.17 seconds per token of sequence length,
so a 4-token prompt extended by 12 tokens runs about 114 token-forwards,
or roughly 19 seconds.

Adding a key/value cache is the next performance milestone; it replaces
that growing forward pass with a single-token step and is what makes
generation practical.

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

### End-to-end parity

The opt-in parity test drives the whole pipeline — text in, tokenizer,
model, greedy loop, tokenizer out — and compares it with the same
pipeline in Hugging Face.

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
GPT2CPP context length:      1,024
cases:                           3
generated tokens compared:      32
mismatches:                      0
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
