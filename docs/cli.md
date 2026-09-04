# Command-line generation

The `gpt2` executable connects the checkpoint loader, tokenizer, model,
generation loop and key/value cache into one command.

```bash
./build-release/gpt2 \
  --checkpoint models/gpt2-small-fp32.bin \
  --vocab path/to/vocab.json \
  --merges path/to/merges.txt \
  --prompt "The capital of France is" \
  --max-new-tokens 20
```

Greedy decoding is the default. Standard output contains only the prompt
and its continuation, which makes it easy to redirect the generated text
to another program or file. Diagnostics go to standard error, and a
failure returns a non-zero exit status.

Run `gpt2 --help` for the complete option list.

## Sampling

Pass `--sample` to enable temperature, top-k and top-p sampling:

```bash
./build-release/gpt2 \
  --checkpoint models/gpt2-small-fp32.bin \
  --vocab path/to/vocab.json \
  --merges path/to/merges.txt \
  --prompt "Once upon a time" \
  --max-new-tokens 40 \
  --sample \
  --temperature 0.8 \
  --top-k 40 \
  --top-p 0.95 \
  --seed 42
```

The default seed is 42. Supplying the same seed and options reproduces
the same random draws on the same build and platform. Sampling controls
without `--sample` are rejected so that a mistyped command cannot
silently fall back to greedy decoding.

## Cache control

The key/value cache is enabled by default. `--no-cache` selects the
slower full-sequence path for debugging and performance comparisons; it
does not change the intended generated tokens.

## Asset checks

The command validates each layer of the input pipeline:

- required options must appear exactly once
- numeric options must be complete, finite values in their valid ranges
- the prompt must not be empty
- tokenizer assets and the checkpoint must pass their format checks
- tokenizer and model vocabulary sizes must match
- encoded prompts must fit inside the checkpoint context window

The model checkpoint and tokenizer files remain ignored by Git because
they are downloaded or generated artifacts rather than source code.

## Tests

`cli_unit` checks help output, missing and duplicate arguments, unknown
options, malformed numbers, sampling validation, empty prompts and asset
errors without requiring a downloaded model.

When `GPT2_ENABLE_GENERATION_PARITY=ON`, the CLI integration tests also
load GPT-2 Small and exercise both greedy and sampled generation from a
real prompt through decoded output using the public executable.
