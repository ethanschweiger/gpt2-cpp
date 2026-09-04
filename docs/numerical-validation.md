# Numerical Validation

The opt-in parity test compares GPT2CPP with the pretrained Hugging Face
GPT-2 Small model using the raw token IDs for `Hello, world!`:

```text
[15496, 11, 995, 0]
```

Using token IDs directly keeps tokenizer behavior outside this test. The four
positions still exercise token embeddings, position embeddings, causal
self-attention, all 12 transformer blocks, the final layer normalization, and
the tied language-model projection.

## Reproducibility controls

- Checkpoint: `models/gpt2-small-fp32.bin`
- Checkpoint SHA-256:
  `c506385a3b29873ef9148a8a5b672b66711a01c2180865ae46449f74cac0d6dc`
- Hugging Face model: `openai-community/gpt2`
- Hugging Face revision:
  `607a30d783dfa663caf39e06633721c8d4cfcd7e`
- CPU FP32 inference in evaluation mode
- Eager attention with the KV cache disabled
- One PyTorch compute thread and deterministic algorithms
- Offline model loading

The model files remain under the ignored `models/` directory and are not
committed to Git.

## One-time model setup

Create the virtual environment and install the exporter dependencies if they
are not already present:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-export.txt
```

Download only the files needed for the pinned GPT-2 revision:

```bash
HF_HOME="$PWD/models/huggingface-cache" \
  .venv/bin/hf download openai-community/gpt2 \
  config.json generation_config.json model.safetensors \
  --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e
```

Export that exact local snapshot into the GPT2CPP checkpoint format:

```bash
.venv/bin/python tools/export_gpt2.py \
  --model "$PWD/models/huggingface-cache/hub/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e" \
  --output "$PWD/models/gpt2-small-fp32.bin" \
  --local-files-only
```

## Running the test

Configure a Release build with the repository virtual environment:

```bash
cmake -S . -B build-parity \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPT2_WARNINGS_AS_ERRORS=ON \
  -DGPT2_ENABLE_GPT2_SMALL_PARITY=ON \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
```

Then build and run only the real-model test:

```bash
cmake --build build-parity --parallel
ctest --test-dir build-parity -L real-model --output-on-failure -V
```

If the ignored model assets are stored elsewhere, configure their absolute
paths with `-DGPT2_SMALL_CHECKPOINT=...` and
`-DGPT2_SMALL_HF_REFERENCE=...`.

The test compares all `4 * 50,257 = 201,028` logits. It requires matching
shapes, finite values, maximum absolute error no greater than `5e-4`, root
mean square error no greater than `1e-4`, and identical top-1 predictions at
all four positions.

## Baseline result

The initial AppleClang Release validation produced:

```text
Compared logits:             201,028
Maximum absolute error:   1.068115234e-4
Mean absolute error:      1.894048389e-5
Root mean square error:   2.302694366e-5
Top-1 agreement:                     4/4
```

Small nonzero differences are expected because the C++ loops and PyTorch
kernels accumulate FP32 products in different orders. The close full-logit
agreement demonstrates that the exported weights, tensor layout, layer
assembly, causal attention, and tied output projection agree with the
reference implementation.
