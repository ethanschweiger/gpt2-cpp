# GPT2CPP

GPT2CPP is a C++20 CPU inference engine for GPT-2 Small (124,439,808
parameters). It loads a compact binary checkpoint, runs the transformer
forward pass and generates text from a prompt. It supports both FP32 and
weights-only int8 checkpoints.

The code covers the complete inference pipeline, from tensor storage and
transformer layers to tokenization, generation, KV caching and quantization.
Each part is tested independently and measured in the benchmark suite.

## Features

- C++20, CMake and AppleClang/Clang/GCC builds
- GPT-2 byte-level BPE tokenization
- GPT-2 Small checkpoint export from a pinned Hugging Face/PyTorch snapshot
- Greedy generation and temperature/top-k/top-p sampling
- Deterministic seeded sampling
- Incremental KV-cache decoding, with an uncached reference path for parity
- FP32, int8-transformer, and int8-transformer-plus-embedding checkpoints
- Binary checkpoint validation, including version, endianness, tensor shape,
  dtype and payload checks
- Unit, integration, numerical-parity, sanitizer and benchmark workflows

## Quick start

```bash
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug \
  -DGPT2_WARNINGS_AS_ERRORS=ON
cmake --build build-debug --parallel
ctest --test-dir build-debug --output-on-failure
```

The normal build does not download model files. Place checkpoints under
`models/` (ignored by Git) before running inference.

## Generate text

```bash
HF_SNAPSHOT=models/huggingface-cache/hub/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e
./build-debug/gpt2 \
  --checkpoint models/gpt2-small-fp32.bin \
  --vocab "$HF_SNAPSHOT/vocab.json" \
  --merges "$HF_SNAPSHOT/merges.txt" \
  --prompt "The capital of France is" \
  --max-new-tokens 24
```

Sampling is opt-in and reproducible with a seed:

```bash
./build-debug/gpt2 \
  --checkpoint models/gpt2-small-fp32.bin \
  --vocab "$HF_SNAPSHOT/vocab.json" \
  --merges "$HF_SNAPSHOT/merges.txt" \
  --prompt "Once upon a time" \
  --max-new-tokens 24 \
  --temperature 0.8 --top-k 40 --top-p 0.95 --seed 42
```

Use `--no-cache` to exercise the reference decoder. The default cached path
and the uncached path are required to produce identical token IDs and stop
reasons in the parity tests.

## Reproducible benchmarks

The generation benchmark is opt-in so ordinary builds stay small:

```bash
cmake -S . -B build-benchmark \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPT2_WARNINGS_AS_ERRORS=ON \
  -DGPT2_BUILD_BENCHMARKS=ON
cmake --build build-benchmark --parallel
ctest --test-dir build-benchmark --output-on-failure
```

Run the full cached-vs-uncached baseline with:

```bash
./build-benchmark/gpt2_generation_benchmark \
  --checkpoint models/gpt2-small-fp32.bin \
  --json benchmarks/results/gpt2-small-release.json
```

The harness alternates execution order, discards warm-ups, checks token
agreement, reports median latency/tokens-per-second, and records raw trials
and environment metadata. See [docs/benchmarks.md](docs/benchmarks.md).

## Quantization results

The committed GPT-2 Small benchmark used a pinned Hugging Face revision, a
658-token evaluation corpus, two warm-up pairs and five measured generation
pairs on an Apple M3.

| Configuration | Checkpoint | Peak RSS | Cached tok/s | Perplexity | Top-1 vs FP32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | 497.8 MB | 632.0 MB | 7.58 | 27.39 | — |
| int8 transformer | 243.3 MB | 364.0 MB | 7.48 | 27.43 | 98.02% |
| int8 transformer + wte | 127.7 MB | 282.8 MB | 7.31 | 28.22 | 80.40% |

These are weights-only int8 checkpoints with inline dequantization, not a
specialized int8 GEMM kernel. They substantially reduce checkpoint and memory
footprint, while the current implementation is slightly slower than FP32;
that tradeoff is measured rather than assumed. Full accuracy, hashes,
uncached throughput and methodology are in
[docs/quantization.md](docs/quantization.md) and
[benchmarks/results/quantization-benchmark.json](benchmarks/results/quantization-benchmark.json).

## How it works

The runtime is split into a few straightforward layers:

1. `Tensor` and tensor operations provide shape-safe contiguous storage and
   matrix primitives.
2. Checkpoint code validates and loads FP32/int8 tensor records.
3. Embedding, layer normalization, GELU, linear, attention and transformer
   modules implement GPT-2's forward pass.
4. `Gpt2Model` composes the 12-layer model and owns KV-cache execution.
5. The tokenizer and generation APIs expose text and token-level inference.

The focused design notes in [docs/](docs/) cover
[generation.md](docs/generation.md), [checkpoint-format.md](docs/checkpoint-format.md),
and [quantization.md](docs/quantization.md).

## Testing and reproducibility

The default CTest suite covers tensor operations, checkpoint loading,
quantized tensors, layers, attention, transformer blocks, model execution,
generation, KV-cache parity, sampling, tokenization, the CLI and exporter
validation. Offline Hugging Face parity tests are enabled explicitly with
the corresponding `GPT2_ENABLE_*` CMake options.

For clean-room export and parity setup, see
[docs/numerical-validation.md](docs/numerical-validation.md). Model files,
build directories and Python environments are intentionally excluded from
version control; source code, tests, benchmark methodology and measured JSON
results are tracked.

## Project highlights

- Built a C++20 GPT-2 Small inference engine from tensor primitives through
  tokenizer and generation APIs.
- Implemented incremental KV caching and verified that cached and uncached
  decoding produce identical token sequences.
- Added weights-only per-channel int8 export, loading and runtime support,
  then measured its accuracy, memory and throughput tradeoffs.
- Reduced the GPT-2 Small checkpoint from 497.8 MB FP32 to 127.7 MB with full
  int8 storage while documenting the associated accuracy and runtime costs.
