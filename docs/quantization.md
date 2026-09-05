# Quantization

`tools/export_gpt2.py` can export GPT-2 with five of its weight matrices
stored as int8 instead of FP32: the four per-layer linear projections
(`c_attn`, `c_proj` in attention, `c_fc` and `c_proj` in the MLP) and,
optionally, the tied token embedding / LM-head weight (`wte`). See
[Checkpoint Format](checkpoint-format.md) for the binary encoding this
produces and the invariants a checkpoint must satisfy.

## Why weights only, and why measure speed rather than assume it

Storing a weight as int8 shrinks the checkpoint on disk by 4x for that
tensor, unconditionally. It does **not** by itself make inference
faster, and which of two very different things happens next changes
the story completely:

- **Dequantize once at load time.** The checkpoint is smaller, but the
  model runs on a full FP32 copy again immediately afterward — no
  runtime memory or cache benefit at all, because the int8 form never
  outlives the loader.
- **Keep weights as int8 in memory, dequantizing on the fly inside
  each matmul.** Runtime memory drops with the checkpoint. But without
  a real int8 kernel — one written to operate on packed int8 data
  directly, typically with SIMD — a naive per-element dequantize step
  added to the existing FP32 inner loop is *extra* work per multiply,
  not less, and can easily end up slower.

This milestone stores weights as int8 in memory and dequantizes as
each one is used, matching the second bullet — see
[Generation](generation.md) and [Benchmarking](benchmarks.md) for how
the model consumes them and what that costs. Precisely because that
extra division-and-cast is real work, this project measures its
runtime cost rather than assuming compression implies speed; the
result is reported alongside accuracy and size, not asserted in
advance.

**Activations, the residual stream, attention scores, softmax,
LayerNorm, biases and every other small tensor stay FP32.** Nothing
about this milestone touches them.

## The scheme

Each quantized tensor is split into channels along one axis. Every
channel `i` gets its own scale,

```text
s_i = max(|W_i|) / 127
```

and its values quantize as

```text
Q_i = round(W_i / s_i),   Q_i in [-127, 127]
```

recovered approximately as `W_i ≈ s_i * Q_i`. A channel that is entirely
zero gets scale `0` and quantizes to all zero rather than dividing by
zero. `-128` has no representation under this scheme; nothing this
exporter writes can produce it, and the C++ loader
([Checkpoint Format](checkpoint-format.md)) rejects it if it ever
appeared regardless.

Per-channel scaling is what makes this tolerable at all: one scale for
an entire matrix would let a single outlier in one output channel
drag down the precision of every other channel sharing that scale.
With a separate scale per channel, an outlier costs precision only in
its own channel.

### Which axis is a channel

Four of the five tensors are `[in, out]` matrices — the same
convention `gpt2::linear` uses — so their channels are their output
features, the last axis. The tied embedding is different. `wte` is
never passed through `linear`; it is read row-wise for embedding
lookup, and for the tied LM-head projection each row is dotted whole
against the hidden state to produce one logit. Its output channels —
one per vocabulary word — are therefore its **rows**, the first axis,
not its columns.

This distinction cannot be inferred generically from shape: attention's
`c_proj` is 768×768, a square matrix for GPT-2 Small, so "the scale's
length matches one of the tensor's dimensions" alone cannot tell you
which one is meant. The checkpoint format deliberately does not try;
which axis applies is a property of what the tensor represents, fixed
per tensor name in the exporter (and, in the next milestone, in the
model-loading code that reads it back).

## Usage

```bash
.venv/bin/python tools/export_gpt2.py \
  --model "$PWD/models/huggingface-cache/hub/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e" \
  --output models/gpt2-small-int8-transformer.bin \
  --local-files-only \
  --quantize
```

Add `--quantize-tied-embedding` to also quantize `wte` (it has no
effect without `--quantize`):

```bash
.venv/bin/python tools/export_gpt2.py \
  --model "$PWD/models/huggingface-cache/hub/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e" \
  --output models/gpt2-small-int8-full.bin \
  --local-files-only \
  --quantize --quantize-tied-embedding
```

`collect_tensor_records`/`export_model` (`tools/export_gpt2.py`) accept
the same two flags as independent keyword arguments for programmatic
use, and `quantize_per_channel` is exposed directly for testing or
reuse.

## Running quantized weights

Three C++ operations dequantize as they compute rather than up front,
matching the "why measure speed rather than assume it" design above —
each is a drop-in sibling of an existing, already-tested operation:

| existing (FP32) | quantized sibling | lives in |
| --- | --- | --- |
| `matmul` | `quantized_matmul` | `tensor_ops.h` |
| `linear` | `quantized_linear` | `layers.h` |
| `embedding_lookup` | `quantized_embedding_lookup` | `layers.h` |

Each is verified two ways: hand-computed expected values (the same
style the FP32 originals' own tests use), and a direct cross-check
against calling the *original* FP32 operation on an explicitly
dequantized copy of the same weight — so a quantized operation's
result is checked against the un-quantized one it is standing in for,
not only against its own arithmetic restated.

`Gpt2Model` does not yet call any of these. Nothing in the model
currently inspects whether a checkpoint tensor is FP32 or int8 — that
dispatch, and running the two real quantized checkpoints below through
it end to end, is the next step.

## Real GPT-2 Small checkpoint sizes

Measured against the same pinned revision used throughout this project
(`607a30d783dfa663caf39e06633721c8d4cfcd7e`; see
[Numerical Validation](numerical-validation.md)):

```text
config 1 - FP32 baseline:              497,770,048 bytes (100.0%)
config 2 - int8 transformer, FP32 wte: 243,301,944 bytes  (48.9%)
config 3 - int8 transformer and wte:   127,710,918 bytes  (25.7%)
```

`wte` alone accounts for about 38.6 million of GPT-2 Small's 124.4
million parameters — nearly a third of the model — which is why config
3 is roughly another 2x smaller than config 2 rather than a marginal
improvement on it. `parameter_count` in `ExportSummary` is identical
across all three exports (124,439,808): quantizing changes how a
weight is stored, never what model it represents, and a per-channel
scale is quantization metadata rather than a model parameter, so it is
excluded from that count.

Accuracy (mean and maximum logit error, top-1 and top-5 agreement
against the FP32 baseline) and runtime speed and memory are measured
separately, once the model-loading side of this milestone can run
these checkpoints — see [Benchmarking](benchmarks.md).

## Testing

- `tests/checkpoint_writer_test.py` and `tests/checkpoint_test.cpp`
  cover the int8 wire format itself (see
  [Checkpoint Format](checkpoint-format.md)).
- `tests/export_gpt2_test.py`'s `QuantizePerChannelTest` checks
  `quantize_per_channel` against an independently written, unvectorized
  reference implementation, a channel's own maximum landing on exactly
  ±127, an all-zero channel avoiding division by zero, and every
  dequantized value staying within half a quantization step of the
  original — all on synthetic arrays, not a real model.
- `QuantizedExportTest` (same file) checks the exporter's tensor
  selection and axis choice on a small fake model: the four
  transformer weights quantize with their output-channel axis, `wte`
  quantizes with its vocabulary axis only when requested, and
  everything else (biases, LayerNorm, position embeddings) never
  changes dtype.
- `tests/hf_export_integration_test.py` runs the same checks against a
  real (though untrained) `GPT2LMHeadModel`, with signed,
  varied-magnitude synthetic weights, and additionally confirms every
  dequantized element of every quantized tensor reproduces its FP32
  source within half its channel's quantization step — real PyTorch
  tensors and real per-channel math, not the lighter fake-tensor
  stand-in the unit tests above use.
- `tests/tensor_ops_test.cpp` and `tests/layers_test.cpp` cover
  `quantized_matmul`, `quantized_linear` and
  `quantized_embedding_lookup`: hand-computed expected values, a match
  against the FP32 operation over an explicitly dequantized copy of
  the same weight, and (for the embedding lookup) that each row is
  dequantized with *its own* row's scale rather than, say, the scale
  at its position in the output — the specific mistake a token order
  that happened to match row order would hide.
