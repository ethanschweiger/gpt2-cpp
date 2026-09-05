# Benchmarking

The generation benchmark measures the public greedy-generation path with
and without the key/value cache. It is designed to produce defensible
project metrics rather than a single convenient timing.

## Build

Benchmarks are opt-in so that normal builds stay small:

```bash
cmake -S . -B build-benchmark \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPT2_WARNINGS_AS_ERRORS=ON \
  -DGPT2_BUILD_BENCHMARKS=ON
cmake --build build-benchmark --parallel
```

Do not publish Debug-build timings. Debug mode intentionally gives up
compiler optimization in exchange for easier diagnosis.

## Quick validation

Use a short run to validate a new build or machine:

```bash
./build-benchmark/gpt2_generation_benchmark \
  --checkpoint models/gpt2-small-fp32.bin \
  --prompt-tokens 4 \
  --new-tokens 2 \
  --warmups 1 \
  --trials 2 \
  --json build-benchmark/quick.json
```

## Baseline run

The default workload is an 8-token prompt extended by 24 tokens, with one
unmeasured warm-up pair and three measured pairs:

```bash
./build-benchmark/gpt2_generation_benchmark \
  --checkpoint models/gpt2-small-fp32.bin \
  --json build-benchmark/gpt2-small-generation.json
```

The full run is deliberately slower because every measured pair includes
the uncached reference path. The benchmark prints progress while it runs.

## Methodology

Each pair generates from the same deterministic synthetic token prompt:

1. One path runs with the key/value cache and one without it.
2. Their order alternates between pairs to reduce systematic thermal and
   scheduling bias.
3. Generated token IDs and stop reasons must match exactly.
4. Warm-up pairs are executed but excluded from the result.
5. The median of the measured trials is used for total latency,
   tokens-per-second and cache speedup.

The measurement includes cache allocation and the prompt prefill because
it times the public `generate_greedy` operation a CLI user actually runs.
It excludes checkpoint loading from generation latency and reports that
time separately. Checkpoint-load timing can be affected by the operating
system's file cache, so treat it as an observed load time rather than a
guaranteed cold-start measurement.

This is a single-threaded CPU inference engine. The reported hardware
thread count describes the machine; it does not mean the benchmark uses
that many worker threads.

## Baseline result

Recorded with `tools/record_baseline.py`, which wraps the benchmark's own
JSON with the checkpoint's SHA-256 and the machine it ran on:

```bash
.venv/bin/python tools/record_baseline.py \
  --runner build-benchmark/gpt2_generation_benchmark \
  --checkpoint models/gpt2-small-fp32.bin \
  --output benchmarks/results/gpt2-small-release.json \
  --warmups 2 \
  --trials 7
```

### Reproducibility controls

- Checkpoint: `models/gpt2-small-fp32.bin`
- Checkpoint SHA-256:
  `c506385a3b29873ef9148a8a5b672b66711a01c2180865ae46449f74cac0d6dc`
- CPU: Apple M3
- OS: macOS 15.7.4 (24G517)
- Compiler: Clang 17.0.0 (clang-1700.0.13.5), Release build
- Workload: an 8-token synthetic prompt extended by 24 tokens
- 2 unmeasured warm-up pairs, 7 measured pairs

### Result

```text
cached median:      3.223 s  (7.45 tokens/s)
uncached median:    38.714 s (0.62 tokens/s)
cache speedup:      12.01x
generated tokens agree between paths: yes (all 7 pairs)
```

Every measured trial is within 1.2% of its path's median — 3.20–3.24 s
cached, 38.66–39.11 s uncached — so this is not an artifact of one lucky
run. The raw per-trial timings, full environment, and checkpoint SHA-256
are committed at
[`benchmarks/results/gpt2-small-release.json`](../benchmarks/results/gpt2-small-release.json).

This machine and workload only: a longer prompt or a different processor
will change both numbers, though the growing gap between them as context
length increases is architectural (see
[Interpreting the result](#interpreting-the-result)), not specific to this run.

This replaces an earlier baseline (cached 3.326 s, uncached 48.389 s,
speedup 14.55x) recorded before
[the last-token-only projection](generation.md#the-last-token-only-projection)
existed. That change cut the uncached median by 20% — it recomputes the
whole sequence at every step, so avoiding wasted projection work at
every one of those steps compounds — while the cached path, which is
already just one new token per step, only benefited on its single
prefill call and moved by about 3%, within the run's own noise. The
speedup ratio *falling* from 14.55x to 12.01x reflects that directly:
the reference path got faster and the one users actually run barely
changed, so the gap between them narrowed. A shrinking ratio here is
what a targeted fix to the slower path is supposed to do — it is not a
regression in the cached path, which is what to check instead.

## JSON output

`--json` writes:

- platform, architecture, compiler, build type, C++ standard and hardware
  thread count
- checkpoint byte size and GPT-2 model dimensions
- exact synthetic prompt token IDs plus generation, warm-up and trial counts
- every cached and uncached trial duration
- median latency and tokens per second
- cache speedup and token-agreement status

The JSON file makes later README tables traceable to raw measurements.
`tools/record_baseline.py` (used above) runs the benchmark and augments
this JSON with a `provenance` object — CPU model, operating-system
version, checkpoint SHA-256, and a UTC timestamp — none of which the
benchmark binary itself can portably determine. It shells out to
`sysctl`/`sw_vers` on macOS and `/proc/cpuinfo`/`/etc/os-release` on
Linux, falling back to Python's `platform` module elsewhere.

## Interpreting the result

The speedup applies to the specified workload and machine, not to every
prompt length or processor. Cached decoding still attends over all prior
tokens, so its work grows with context length; it avoids recomputing the
earlier tokens' projections and MLP layers.

[Profiling](profiling.md) breaks down where this run's time actually
went, both cached and uncached, and names the concrete target for the
next optimization pass.

[Quantization](quantization.md) covers a different kind of optimization:
storing four of the per-layer weight matrices, and optionally the tied
embedding, as int8 instead of FP32. Its checkpoint-size results are
already measured; the accuracy and speed sides of that comparison are
recorded here once the model can run those checkpoints.
