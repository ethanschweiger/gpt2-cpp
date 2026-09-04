from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import random
import shutil
import subprocess
import tempfile


EXPECTED_VOCABULARY_SHA256 = (
    "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783"
)
EXPECTED_MERGES_SHA256 = (
    "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5"
)
EXPECTED_VOCABULARY_SIZE = 50257
END_OF_TEXT = "<|endoftext|>"

RANDOM_SEED = 20260904
RANDOM_TEXT_SAMPLES = 3000
RANDOM_DECODE_SAMPLES = 2000
CODE_POINT_STRIDE = 257
REPORTED_MISMATCH_LIMIT = 10

PROSE = (
    "The quick brown fox jumps over the lazy dog. "
    "In 1948, a Bell Labs paper founded information theory; "
    "it didn't take long for the idea to spread. "
    "Prices rose 12.5% year-over-year, reaching $1,234.56 per unit -- "
    "an increase nobody's forecast had predicted.\n\n"
    "\tIndented lines, trailing spaces,   doubled  spaces, and\n"
    "CRLF-style breaks all have to survive a round trip.\r\n"
    "Café naïve résumé — "
    "日本語のテキスト — "
    "\U0001f642\U0001f30d."
)

EDGE_CASES = (
    "",
    " ",
    "  ",
    "   ",
    "\n",
    "\n\n",
    "\n\n\n",
    "\t",
    "\r\n",
    "\v",
    "\f",
    "a",
    "A",
    "0",
    "a b",
    "a  b",
    "a   b",
    "   a",
    "abc   ",
    "  \n  ",
    "hello world",
    "Hello, world!",
    "don't",
    "Don'T",
    "DON'T",
    "it's 'S 'll 'LL 're 'VE 'm 'd",
    "'s",
    "'",
    "''",
    "'x",
    "123 4567 89",
    "3.14159",
    "-42",
    "1,000,000",
    "café",
    "naïve résumé",
    "日本語のテキスト",
    "\U0001f642\U0001f643",
    "é",
    " no-break space",
    " line separator",
    " paragraph separator",
    "　ideographic space",
    " ogham space",
    " four-per-em space",
    " narrow no-break space",
    "mixed ½ fractions Ⅰ",
    "AB­CD",
    "zero​width",
    "",
    END_OF_TEXT,
    " " + END_OF_TEXT + " ",
    "a" + END_OF_TEXT + "b",
    END_OF_TEXT + END_OF_TEXT,
    END_OF_TEXT + "\n" + END_OF_TEXT,
    "<|endoftext|",
    "|endoftext|>",
    "<|ENDOFTEXT|>",
    "<|endoftext<|endoftext|>",
    "<|endoftext|<|endoftext|>",
    "<|endoftext|><|endoftext",
    "<|end<|endoftext|>oftext|>",
    "<|endoftext|>" * 4,
    "supercalifragilisticexpialidocious",
    "a" * 64,
    " " * 16,
    "\n" * 16,
    PROSE,
)

RANDOM_ALPHABET = (
    list("abcdefghijklmnopqrstuvwxyz")
    + list("ABCXYZ")
    + list("0123456789")
    + [" "] * 8
    + ["\t", "\n", "\r", "\v", "\f"]
    + list("'\".,!?;:%-_/\\()[]{}<>|&#@$*+=~^`")
    + [" ", " ", "　", " ", " ", " "]
    + ["é", "ü", "ß", "ç", "Ł"]
    + ["日", "本", "語", "안", "Ж"]
    + ["א", "ا", "ก", "০", "٣"]
    + ["­", "​", "́", "α", "Ω"]
    + ["Ⅰ", "½", "·", "«", "»", "—"]
    + ["\U0001f642", "\U0001f1fa", "\U0001f1f8", "\U0001d400"]
    + [END_OF_TEXT]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the GPT2CPP tokenizer with the matching Hugging "
            "Face tokenizers."
        )
    )
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--merges", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as asset:
        while chunk := asset.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_to_unicode() -> dict[int, str]:
    """The byte-level alphabet from the original GPT-2 release."""
    printable = (
        list(range(0x21, 0x7F))
        + list(range(0xA1, 0xAD))
        + list(range(0xAE, 0x100))
    )
    mapped = printable[:]
    extra = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            mapped.append(256 + extra)
            extra += 1
    return dict(zip(printable, [chr(point) for point in mapped]))


UNICODE_TO_BYTE = {
    character: byte for byte, character in bytes_to_unicode().items()
}


def build_text_samples() -> list[str]:
    samples: list[str] = list(EDGE_CASES)

    # Every code point in the ranges ordinary text uses, plus a strided
    # sample of the rest, in the contexts where the pre-tokenizer's
    # character classes decide a piece boundary.
    code_points = list(range(0x0000, 0x0300))
    code_points += list(range(0x0300, 0x110000, CODE_POINT_STRIDE))
    for code_point in code_points:
        if 0xD800 <= code_point <= 0xDFFF:
            continue
        character = chr(code_point)
        samples.append(character)
        samples.append("a" + character + "b")
        samples.append("1" + character + "2")
        samples.append(" " + character + " ")
        samples.append(character + character)

    generator = random.Random(RANDOM_SEED)
    for _ in range(RANDOM_TEXT_SAMPLES):
        length = generator.randint(0, 24)
        samples.append(
            "".join(
                generator.choice(RANDOM_ALPHABET) for _ in range(length)
            )
        )

    # Sliding windows over the prose exercise piece boundaries at every
    # offset within real text.
    for start in range(0, len(PROSE) - 24, 7):
        samples.append(PROSE[start:start + 24])

    return samples


def build_decode_samples() -> list[list[int]]:
    generator = random.Random(RANDOM_SEED + 1)
    sequences: list[list[int]] = [
        [],
        [0],
        [EXPECTED_VOCABULARY_SIZE - 1],
        [220, 220, 220],
        [198, 628],
    ]

    for _ in range(RANDOM_DECODE_SAMPLES):
        length = generator.randint(0, 20)
        sequences.append(
            [
                generator.randrange(EXPECTED_VOCABULARY_SIZE)
                for _ in range(length)
            ]
        )

    return sequences


def run_runner(
    runner: Path,
    vocabulary: Path,
    merges: Path,
    requests: list[str],
) -> tuple[list[str], int]:
    with tempfile.TemporaryDirectory() as directory:
        request_path = Path(directory) / "requests.txt"
        answer_path = Path(directory) / "answers.txt"
        request_path.write_text(
            "\n".join(requests) + "\n",
            encoding="ascii",
        )

        completed = subprocess.run(
            (
                str(runner),
                str(vocabulary),
                str(merges),
                str(request_path),
                str(answer_path),
            ),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "C++ tokenizer runner failed:\n" + completed.stderr.strip()
            )

        answers = answer_path.read_text(encoding="ascii").splitlines()

    return answers, int(completed.stdout.strip())


def reference_bytes(tokenizer, token_ids: list[int]) -> bytes:
    """Reverses the byte-level alphabet directly.

    Hugging Face's own decode replaces invalid UTF-8 with U+FFFD, while
    GPT2CPP returns the bytes the tokens carry, so the comparison uses
    the token strings instead.
    """
    if not token_ids:
        return b""

    return bytes(
        UNICODE_TO_BYTE[character]
        for token in tokenizer.convert_ids_to_tokens(token_ids)
        for character in token
    )


def load_reference_tokenizers(directory: Path, vocabulary: Path, merges: Path):
    """Loads both Hugging Face tokenizers from the same two assets
    GPT2CPP reads, so neither side sees a prebuilt tokenizer.json."""
    from transformers import GPT2Tokenizer, GPT2TokenizerFast

    shutil.copyfile(vocabulary, directory / "vocab.json")
    shutil.copyfile(merges, directory / "merges.txt")

    slow = GPT2Tokenizer.from_pretrained(
        str(directory),
        local_files_only=True,
    )
    fast = GPT2TokenizerFast.from_pretrained(
        str(directory),
        local_files_only=True,
    )
    return slow, fast


def main() -> None:
    args = parse_args()

    if not args.runner.is_file():
        raise FileNotFoundError(f"C++ runner not found: {args.runner}")

    digests: dict[str, str] = {}
    for asset, expected in (
        (args.vocabulary, EXPECTED_VOCABULARY_SHA256),
        (args.merges, EXPECTED_MERGES_SHA256),
    ):
        if not asset.is_file():
            raise FileNotFoundError(f"tokenizer asset not found: {asset}")
        digest = sha256(asset)
        digests[asset.name] = digest
        if digest != expected:
            raise AssertionError(
                f"{asset.name} SHA-256 mismatch: "
                f"expected {expected}, received {digest}"
            )

    import tokenizers
    import transformers

    with tempfile.TemporaryDirectory() as directory:
        slow, fast = load_reference_tokenizers(
            Path(directory),
            args.vocabulary,
            args.merges,
        )

    text_samples = build_text_samples()
    decode_samples = build_decode_samples()

    requests = [
        "e " + sample.encode("utf-8").hex() for sample in text_samples
    ]
    requests += [
        ("d " + " ".join(str(token_id) for token_id in sequence)).rstrip()
        for sequence in decode_samples
    ]

    answers, vocabulary_size = run_runner(
        args.runner,
        args.vocabulary,
        args.merges,
        requests,
    )

    if len(answers) != len(requests):
        raise AssertionError(
            "C++ tokenizer runner answered "
            f"{len(answers)} of {len(requests)} requests"
        )

    encoded: list[list[int]] = []
    for index in range(len(text_samples)):
        answer = answers[index]
        if not answer.startswith("i"):
            raise AssertionError(
                f"encode request {index} received a non-encode answer"
            )
        encoded.append([int(field) for field in answer[1:].split()])

    slow_mismatches: list[str] = []
    slow_mismatch_count = 0
    fast_mismatch_count = 0
    reference_disagreements = 0
    round_trip_failures = 0

    for sample, cpp_ids in zip(text_samples, encoded):
        slow_ids = slow.encode(sample)
        fast_ids = fast.encode(sample)
        references_agree = slow_ids == fast_ids
        if not references_agree:
            reference_disagreements += 1

        # GPT2CPP classifies characters with the same Unicode database
        # the slow tokenizer's Python "regex" patterns use, so the slow
        # tokenizer is the reference and the fast one is compared only
        # where the two agree with each other.
        if cpp_ids != slow_ids:
            slow_mismatch_count += 1
            if len(slow_mismatches) < REPORTED_MISMATCH_LIMIT:
                slow_mismatches.append(
                    f"  {sample!r}\n"
                    f"    GPT2CPP:     {cpp_ids}\n"
                    f"    Hugging Face: {slow_ids}"
                )
        if references_agree and cpp_ids != fast_ids:
            fast_mismatch_count += 1

        if reference_bytes(fast, cpp_ids) != sample.encode("utf-8"):
            round_trip_failures += 1

    decode_mismatches: list[str] = []
    decode_mismatch_count = 0
    offset = len(text_samples)
    for index, sequence in enumerate(decode_samples):
        answer = answers[offset + index]
        if not answer.startswith("b"):
            raise AssertionError(
                f"decode request {index} received a non-decode answer"
            )

        cpp_bytes = bytes.fromhex(answer[1:].strip())
        expected_bytes = reference_bytes(fast, sequence)
        if cpp_bytes != expected_bytes:
            decode_mismatch_count += 1
            if len(decode_mismatches) < REPORTED_MISMATCH_LIMIT:
                decode_mismatches.append(
                    f"  {sequence}\n"
                    f"    GPT2CPP:      {cpp_bytes!r}\n"
                    f"    Hugging Face: {expected_bytes!r}"
                )

    print("GPT-2 tokenizer parity")
    print(f"  vocabulary SHA-256: {digests[args.vocabulary.name]}")
    print(f"  merges SHA-256: {digests[args.merges.name]}")
    print(f"  Transformers: {transformers.__version__}")
    print(f"  Tokenizers: {tokenizers.__version__}")
    print(f"  GPT2CPP vocabulary size: {vocabulary_size:,}")
    print(f"  encode samples: {len(text_samples):,}")
    print(f"  decode samples: {len(decode_samples):,}")
    print(f"  encode mismatches: {slow_mismatch_count}")
    print(f"  decode mismatches: {decode_mismatch_count}")
    print(f"  round-trip failures: {round_trip_failures}")
    print(
        "  fast-tokenizer mismatches where both references agree: "
        f"{fast_mismatch_count}"
    )
    print(
        "  samples where the two Hugging Face tokenizers disagree: "
        f"{reference_disagreements}"
    )

    if vocabulary_size != EXPECTED_VOCABULARY_SIZE:
        raise AssertionError(
            "GPT2CPP vocabulary size mismatch: "
            f"expected {EXPECTED_VOCABULARY_SIZE}, "
            f"received {vocabulary_size}"
        )
    if slow_mismatch_count != 0:
        raise AssertionError(
            f"{slow_mismatch_count} encoding(s) differ from Hugging "
            "Face:\n" + "\n".join(slow_mismatches)
        )
    if fast_mismatch_count != 0:
        raise AssertionError(
            f"{fast_mismatch_count} encoding(s) differ from the fast "
            "Hugging Face tokenizer"
        )
    if decode_mismatch_count != 0:
        raise AssertionError(
            f"{decode_mismatch_count} decoding(s) differ from Hugging "
            "Face:\n" + "\n".join(decode_mismatches)
        )
    if round_trip_failures != 0:
        raise AssertionError(
            f"{round_trip_failures} sample(s) did not round trip"
        )


if __name__ == "__main__":
    main()
