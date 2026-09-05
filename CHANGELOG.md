# Changelog

## [0.1.0] - 2026-09-04

This is the first complete GPT2CPP milestone: a small, testable C++20 CPU
inference engine that can load, run and generate from GPT-2 Small
checkpoints.

### Added

- Contiguous tensor storage and shape-aware tensor operations
- Validated binary checkpoint format with FP32 and int8 tensor records
- GPT-2 Small embeddings, layer normalization, GELU, linear layers,
  attention and transformer blocks
- GPT-2 byte-level BPE tokenizer and text generation CLI
- Greedy decoding, temperature/top-k/top-p sampling and seeded runs
- Incremental KV-cache decoding with cached/uncached parity checks
- Weights-only per-channel int8 export, loading and inference
- Unit, integration, numerical-parity, sanitizer and benchmark workflows
- Reproducible FP32/int8 accuracy, memory and throughput reports
- Project documentation and a reproducible build/test README

### Verified results

The pinned GPT-2 Small benchmark reduced the checkpoint from 497.8 MB FP32
to 127.7 MB with full int8 storage. On the recorded Apple M3 run, the full
int8 configuration used 282.8 MB peak RSS, achieved 7.31 cached tokens/s and
had 80.40% top-1 agreement with the FP32 baseline over the 658-token
evaluation corpus. The transformer-only int8 configuration retained 98.02%
top-1 agreement and 100% top-5 agreement.

These are weights-only checkpoints with inline dequantization, so the current
int8 path is slightly slower than FP32. The complete raw report is tracked in
[`benchmarks/results/quantization-benchmark.json`](benchmarks/results/quantization-benchmark.json).

### Known limitations

- The runtime is CPU-focused and currently single-threaded.
- The int8 path does not yet use a specialized int8 GEMM/microkernel.
- Model checkpoints and Hugging Face assets are downloaded/generated locally;
  they are intentionally not committed to the repository.
