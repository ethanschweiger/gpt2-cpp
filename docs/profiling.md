# Profiling

Before optimizing the FP32 forward pass, this measures exactly where its
time goes on the real GPT-2 Small model, so the next step targets a
measured cost rather than a guess.

## Method

This machine has neither `perf` nor a working `Instruments`/`xctrace`
install, only the command-line `sample` tool built into macOS. `sample`
periodically captures the call stack of a running process and reports
how many captures landed in each function — a statistical profile, not
an instruction-level trace, but sufficient to rank the handful of
functions that could plausibly dominate a matmul-bound workload.

Two separate profiles were taken from one real run against GPT-2 Small
(the same 8-token-prompt, 24-new-token workload as the recorded
baseline), each started once the run had settled into steady state:

```bash
# process running in the background, checkpoint already loaded
sample <pid> 2 5 -file cached.txt      # during the cached phase
sample <pid> 25 5 -file uncached.txt   # during the uncached phase
```

`generate_greedy`'s cached and uncached paths are visibly distinct
subtrees of the same call graph — `forward(tokens)` versus
`forward(tokens, KvCache&)` — so one profile cleanly separates both.

## Result

| where the time goes | cached samples | cached % | uncached samples | uncached % |
| --- | --- | --- | --- | --- |
| MLP matmuls (`feed_forward`) | 172 / 354 | 48.6% | 2,331 / 4,427 | 52.7% |
| attention matmuls (QKV, output projection, QKᵀ·V) | 88 / 354 | 24.9% | 1,112 / 4,427 | 25.1% |
| tied output projection (`tied_embedding_logits`) | 92 / 354 | 26.0% | 965 / 4,427 | 21.8% |
| GELU / other | 2 / 354 | 0.6% | 19 / 4,427 | 0.4% |

Essentially all of the first two rows is time inside `gpt2::matmul` in
[tensor_ops.cpp](../src/tensor_ops.cpp) — a plain triple-nested loop with
no blocking, no vectorization, and a non-contiguous access pattern on
its right-hand operand (a fixed column stride, so consecutive inner-loop
reads do not share a cache line). That is the concrete target for the
next optimization pass.

## The vocabulary projection is not a rounding error

The tied output projection turns each position's 768-wide hidden state
into a 50,257-wide logit vector. Restated as multiply-adds per token,
one transformer layer costs about 7.08M (QKV projection, attention
output projection, and the two MLP projections) and the projection
itself costs 768 × 50,257 ≈ 38.6M — on its own, roughly **three
attention layers' worth of work**, because GPT-2 Small's vocabulary is
unusually large relative to its hidden width and layer count. Twelve
layers bring the transformer body to about 84.9M multiply-adds, so the
projection's expected share of one token's total work is
38.6 / (84.9 + 38.6) ≈ **31%** — close to both measured columns above.
The two profiles agree with each other, and with this back-of-envelope
figure, because the projection's cost is architectural rather than
implementation-specific: it holds however the token is produced.

## What the profile does and does not justify

The cached-versus-uncached proportions above are nearly identical, and
that is itself informative: the ~15x cache speedup is not concentrated
in any one of these three categories. It comes from the key/value cache
already avoiding exactly the redundant work its name promises — the
uncached path recomputes the whole transformer stack, projection
included, for every already-seen token at every step, while the cached
path processes only the one new token per step (after a one-time
prefill). There is no fourth category hiding a disproportionate,
independent win.

One real, bounded optimization fell out of this reading:
`generate_greedy` and `generate_sampled` only ever read the final row
of whatever logits `Gpt2Model::forward` returns — see `last_row` in
[generation.cpp](../src/generation.cpp) — so computing the projection
for every other position was pure waste from generation's point of
view. It was not waste from every caller's point of view, though: the
model unit test and the 201,028-logit Hugging Face parity test both
depend on `forward` returning every position's logits, so this could
not become `forward`'s new default. `Gpt2Model::forward_last_token_logits`
is the generation-only opt-in path instead — see
[Generation](generation.md#the-last-token-only-projection) for the API
and the tests that prove its last row is bit-identical to `forward`'s.

For the uncached path specifically, restricting the projection to the
final position removes work that scaled with the whole sequence length
and contributed nothing: at the benchmark's longest step (a 31-token
sequence), the projection's share of that step's cost was estimated to
fall from ≈31% to a small constant term, an estimated ~30% reduction in
that step's own cost. For the cached path, the same change only touches
the one-time prefill call — real, but small next to 24 subsequent
single-token steps at this workload's scale; it matters more for a
long-prompt, few-token workload than for this one. The recorded
baseline in [Benchmarking](benchmarks.md#baseline-result) measures both
paths after this change.
