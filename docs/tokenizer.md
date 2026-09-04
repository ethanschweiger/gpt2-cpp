# Tokenizer

`gpt2::Gpt2Tokenizer` implements GPT-2's byte-level byte-pair encoding
from the two assets OpenAI published with the model:

- `vocab.json`, a flat object mapping each token string to its ID
- `merges.txt`, the ranked merge rules

```cpp
#include "gpt2/tokenizer.h"

const gpt2::Gpt2Tokenizer tokenizer = gpt2::Gpt2Tokenizer::load(
    "models/gpt2-vocab.json",
    "models/gpt2-merges.txt"
);

const std::vector<std::size_t> ids = tokenizer.encode("Hello, world!");
// {15496, 11, 995, 0}

const std::string text = tokenizer.decode(ids);
// "Hello, world!"
```

`TokenId` is `std::size_t`, which is what `Gpt2Model::forward` accepts,
so encoded text feeds the model directly.

## Encoding

Encoding runs four stages.

**1. End-of-text splitting.** `<|endoftext|>` cannot be produced by the
merge rules, so `encode` first splits the input on that literal and emits
its ID (50256 for GPT-2) between the surrounding segments. The marker is
matched case-sensitively and only in full: `<|endoftext|` is ordinary
text.

**2. Pre-tokenization.** Each segment is split with GPT-2's pattern:

```text
's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

GPT2CPP has no regular-expression dependency, so `src/tokenizer.cpp`
scans the alternatives in the same order a backtracking engine would try
them. Two details of that order are easy to get wrong and are pinned down
by unit tests:

- The optional leading space in ` ?\p{L}+` is taken only when a class
  member follows it, so `"a b"` splits as `"a"` and `" b"`.
- `\s+(?!\S)` matches a whitespace run whole only at the end of the
  input. Elsewhere it gives back the final character so the following
  ` ?\p{L}+` can claim it, which is why `"   a"` splits as `"  "` and
  `" a"` rather than `"   "` and `"a"`.

The `\p{L}`, `\p{N}` and `\s` character classes come from
`src/unicode_ranges.inc`, a generated table of code-point ranges. See
[Regenerating the Unicode tables](#regenerating-the-unicode-tables).

**3. Byte-level mapping.** Each piece is mapped byte by byte onto
printable code points: bytes `0x21`-`0x7E`, `0xA1`-`0xAC` and
`0xAE`-`0xFF` map to themselves, and the remaining 68 bytes map to
`U+0100` upward in increasing byte order. A space therefore becomes
`U+0120` and a newline becomes `U+010A`.

**4. Byte-pair merging.** The mapped piece starts as one symbol per
character. The lowest-ranked adjacent pair that has a merge rule is
merged everywhere it occurs, scanning left to right without overlap,
until no rule applies. `"aaa"` with a rule for `a a` therefore becomes
`"aa"` and `"a"`, not `"a"` and `"aa"`.

`encode` throws `std::invalid_argument` when the input is not valid
UTF-8. Overlong sequences, surrogates and code points above `U+10FFFF`
are rejected.

Merging is written for clarity rather than speed: every round rescans
the piece for the best-ranked pair, so the cost grows quadratically in
the length of a single piece. Pre-tokenization keeps ordinary pieces to
a word or a short run of punctuation, so this does not matter in
practice; a ranked queue and a per-piece cache belong with the
performance work rather than here.

## Decoding

`decode` concatenates the token strings and reverses the byte-level
mapping, so it returns the bytes the tokens carry. A complete sequence
round trips exactly. A truncated one — the first half of an emoji, say —
returns an incomplete UTF-8 sequence rather than a replacement
character, which leaves the choice of error handling to the caller.
Hugging Face's `decode` differs here: it substitutes `U+FFFD`.

`decode` throws `std::out_of_range` for a token ID outside the
vocabulary.

## What the loader validates

`load` rejects assets that cannot produce a working tokenizer rather
than failing later during encoding:

- the vocabulary must be a JSON object of string keys and non-negative
  integer values, with no duplicates and no trailing content
- token IDs must cover `0` through `size - 1` exactly once
- every token must consist only of byte-level alphabet characters
- all 256 alphabet characters must be present as single-character
  tokens, which is what makes `encode` total for valid UTF-8
- every merge rule must be two non-empty symbols separated by one space,
  must not repeat an earlier rule, and must produce a token that is in
  the vocabulary
- `<|endoftext|>` must be present

A leading `#version:` line in `merges.txt` is skipped. Only that line is
skipped: GPT-2's own merge rules include entries such as `# #` and
`## ##`, so a blanket "drop the first line" rule would silently discard
a real rule from a file without a header. Blank lines are ignored and
`\r\n` line endings are accepted.

## Regenerating the Unicode tables

`src/unicode_ranges.inc` is generated from Python's `regex` module,
which is the same engine the Hugging Face slow tokenizer uses:

```bash
.venv/bin/python tools/generate_unicode_ranges.py
```

`--check` fails instead of writing when the committed table is stale,
and runs as the `gpt2_unicode_ranges_current` test.

The table follows whichever Unicode database the installed `regex`
release carries. The Rust engine behind the Hugging Face *fast*
tokenizer carries an older one, so the two classify recently assigned
code points differently — 4,657 of them for `regex` 2026.9.3 against
`tokenizers` 0.23.1, all cases where Python treats a character as a
letter or a number and Rust does not. None of them change any token ID,
because GPT-2's merge rules were learned before those characters were
assigned and so never join them to their neighbours. The parity test
reports how many samples the two Hugging Face tokenizers disagree on so
that this stays visible if a future release changes it.

## Parity testing

The opt-in parity test compares GPT2CPP with both Hugging Face GPT-2
tokenizers, loading all three from the same `vocab.json` and
`merges.txt`.

### Reproducibility controls

- Hugging Face model: `openai-community/gpt2`
- Hugging Face revision:
  `607a30d783dfa663caf39e06633721c8d4cfcd7e`
- `vocab.json` SHA-256:
  `196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783`
- `merges.txt` SHA-256:
  `1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5`
- Offline model loading with a fixed sample seed

The tokenizer assets stay under the ignored `models/` directory and are
not committed to Git.

### One-time asset setup

With the virtual environment from
[Numerical Validation](numerical-validation.md) in place, download the
tokenizer files for the pinned revision:

```bash
HF_HOME="$PWD/models/huggingface-cache" \
  .venv/bin/hf download openai-community/gpt2 \
  vocab.json merges.txt \
  --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e
```

### Running the test

```bash
cmake -S . -B build-tokenizer \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPT2_WARNINGS_AS_ERRORS=ON \
  -DGPT2_ENABLE_TOKENIZER_PARITY=ON \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
```

```bash
cmake --build build-tokenizer --parallel
ctest --test-dir build-tokenizer -L tokenizer --output-on-failure -V
```

If the ignored assets are stored elsewhere, configure their absolute
paths with `-DGPT2_SMALL_VOCABULARY=...` and `-DGPT2_SMALL_MERGES=...`.

### Coverage

The test builds its samples from four sources:

- hand-written edge cases: whitespace runs, contractions, digits,
  accented and multi-byte text, and every position of the end-of-text
  marker
- every code point from `U+0000` to `U+02FF`, plus a strided sample of
  the rest of the space, each in five contexts that make the
  pre-tokenizer's class boundaries observable
- 3,000 seeded random strings over an alphabet of letters, digits,
  punctuation, exotic whitespace, combining marks, scripts, emoji and
  the end-of-text marker
- sliding windows over a paragraph of mixed prose

It also decodes 2,005 seeded random token sequences, comparing against
the byte-level reconstruction of Hugging Face's own token strings so
that truncated multi-byte characters are compared as bytes rather than
through Hugging Face's lossy replacement.

### Baseline result

```text
GPT2CPP vocabulary size:                            50,257
encode samples:                                     28,590
decode samples:                                      2,005
encode mismatches:                                       0
decode mismatches:                                       0
round-trip failures:                                     0
fast-tokenizer mismatches where both references agree:   0
samples where the two Hugging Face tokenizers disagree:  0
```

Unlike the model's forward pass, tokenization is exact: there is no
tolerance to report, only agreement or disagreement.
