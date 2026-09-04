from __future__ import annotations

import argparse
from pathlib import Path
import random
import struct
import subprocess
import tempfile


RANDOM_SEED = 20260905
VOCABULARY_SIZES = (2, 4, 8, 64, 1000, 50257)
TEMPERATURES = (0.1, 0.5, 0.7, 1.0, 1.5, 4.0)
TOP_K_VALUES = (0, 1, 2, 5, 40, 1000)
TOP_P_VALUES = (0.05, 0.3, 0.5, 0.9, 0.92, 0.99, 1.0)
REPEATS = 3

# Which tokens survive the filters is a semantic question and is
# compared exactly. The probabilities themselves are not: GPT2CPP sums
# the softmax denominator in double, so it lands closer to the exact
# answer than a float32 reference does. The reference is therefore
# computed in float64, and the tolerance only has to cover rounding the
# result back to float32, which costs a couple of units in the last
# place.
PROBABILITY_TOLERANCE = 4.0e-7
REPORTED_MISMATCH_LIMIT = 5

# A probability below this cannot survive being stored as float32, so a
# token the reference keeps with a smaller probability is expected to
# read as zero on the C++ side. That is underflow, not a filter
# disagreement, and is counted separately.
FLOAT32_UNDERFLOW = 1.0e-37


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the GPT2CPP sampling filters with the matching "
            "Hugging Face logits warpers."
        )
    )
    parser.add_argument("--runner", type=Path, required=True)
    return parser.parse_args()


def build_cases() -> list[tuple[list[float], float, int, float]]:
    generator = random.Random(RANDOM_SEED)
    cases: list[tuple[list[float], float, int, float]] = []

    # A hand-built case where several scores tie at the top-k boundary,
    # which is where "keep exactly k" and "keep everything at least as
    # high as the k-th score" disagree.
    cases.append(([3.0, 1.0, 1.0, 1.0, 0.0], 1.0, 2, 1.0))
    cases.append(([3.0, 1.0, 1.0, 1.0, 0.0], 1.0, 3, 1.0))
    cases.append(([0.0, 0.0, 0.0, 0.0], 1.0, 2, 1.0))
    cases.append(([5.0, -5.0], 1.0, 0, 0.5))

    for size in VOCABULARY_SIZES:
        for repeat in range(REPEATS):
            scores = [generator.gauss(0.0, 3.0) for _ in range(size)]
            for temperature in TEMPERATURES:
                for top_k in TOP_K_VALUES:
                    for top_p in TOP_P_VALUES:
                        if repeat != 0 and (top_k, top_p) != (0, 1.0):
                            continue
                        cases.append((scores, temperature, top_k, top_p))

    # Quantised scores create ties on purpose, at several vocabulary
    # sizes, so the top-k boundary rule is exercised broadly.
    for size in (8, 64, 1000):
        for _ in range(20):
            scores = [
                float(round(generator.gauss(0.0, 2.0)))
                for _ in range(size)
            ]
            for top_k in (1, 2, 5, 40):
                cases.append((scores, 1.0, top_k, 0.95))

    return cases


def run_runner(
    runner: Path,
    cases: list[tuple[list[float], float, int, float]],
) -> list[list[float]]:
    with tempfile.TemporaryDirectory() as directory:
        case_path = Path(directory) / "cases.txt"
        score_path = Path(directory) / "scores.bin"
        output_path = Path(directory) / "probabilities.bin"

        lines = []
        with score_path.open("wb") as scores_file:
            for scores, temperature, top_k, top_p in cases:
                lines.append(
                    f"{len(scores)} {temperature!r} {top_k} {top_p!r}"
                )
                scores_file.write(
                    struct.pack(f"<{len(scores)}f", *scores)
                )
        case_path.write_text("\n".join(lines) + "\n", encoding="ascii")

        completed = subprocess.run(
            (
                str(runner),
                str(case_path),
                str(score_path),
                str(output_path),
            ),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "C++ sampling runner failed:\n" + completed.stderr.strip()
            )

        raw = output_path.read_bytes()

    answered = int(completed.stdout.strip())
    if answered != len(cases):
        raise AssertionError(
            f"C++ runner answered {answered} of {len(cases)} cases"
        )

    results: list[list[float]] = []
    offset = 0
    for scores, _, _, _ in cases:
        size = len(scores)
        results.append(
            list(struct.unpack_from(f"<{size}f", raw, offset))
        )
        offset += size * 4
    if offset != len(raw):
        raise AssertionError("C++ runner wrote unexpected trailing bytes")

    return results


def reference_probabilities(
    scores: list[float],
    temperature: float,
    top_k: int,
    top_p: float,
    dtype,
):
    import torch
    from transformers.generation.logits_process import (
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )

    empty = torch.zeros((1, 0), dtype=torch.long)
    # The C++ side receives float32 scores, so the reference starts
    # from the same rounded values before widening.
    processed = torch.tensor([scores], dtype=torch.float32).to(dtype)
    processed = TemperatureLogitsWarper(temperature)(empty, processed)
    if top_k != 0:
        processed = TopKLogitsWarper(top_k)(empty, processed)
    if top_p < 1.0:
        processed = TopPLogitsWarper(top_p)(empty, processed)

    return processed.softmax(dim=-1)[0].tolist()


def main() -> None:
    args = parse_args()
    if not args.runner.is_file():
        raise FileNotFoundError(f"C++ runner not found: {args.runner}")

    import tokenizers  # noqa: F401  (pins the reference environment)
    import torch
    import transformers

    torch.set_num_threads(1)

    cases = build_cases()
    actual = run_runner(args.runner, cases)

    mismatches: list[str] = []
    mismatch_count = 0
    support_mismatch_count = 0
    largest_difference = 0.0
    largest_float32_difference = 0.0
    largest_reference_rounding = 0.0
    compared_values = 0
    underflowed = 0
    tie_ambiguous = 0

    for index, (scores, temperature, top_k, top_p) in enumerate(cases):
        expected = reference_probabilities(
            scores,
            temperature,
            top_k,
            top_p,
            torch.float64,
        )
        single = reference_probabilities(
            scores,
            temperature,
            top_k,
            top_p,
            torch.float32,
        )
        received = actual[index]
        compared_values += len(expected)

        # A token the filters removed has probability exactly zero in
        # the reference, so keeping one is always wrong. Dropping one is
        # only wrong when the reference probability was large enough to
        # survive float32.
        over_kept = [
            token
            for token, value in enumerate(expected)
            if value == 0.0 and received[token] > 0.0
        ]
        under_kept = [
            token
            for token, value in enumerate(expected)
            if value > FLOAT32_UNDERFLOW and received[token] == 0.0
        ]
        underflowed += sum(
            1
            for token, value in enumerate(expected)
            if 0.0 < value <= FLOAT32_UNDERFLOW and received[token] == 0.0
        )
        # Tied probabilities make the top-p boundary ambiguous: which
        # of several equally likely tokens survives depends on the sort
        # order, and torch.sort is not stable. The kept count and the
        # kept distribution are still well defined, so that is what gets
        # compared when the sets disagree.
        kept_expected = sorted(value for value in expected if value > 0.0)
        kept_received = sorted(value for value in received if value > 0.0)
        equivalent = len(kept_expected) == len(kept_received) and all(
            abs(left - right) <= PROBABILITY_TOLERANCE
            for left, right in zip(kept_expected, kept_received)
        )

        support_differs = False
        if over_kept or under_kept:
            if equivalent:
                tie_ambiguous += 1
            else:
                support_differs = True
                support_mismatch_count += 1

        if over_kept or under_kept:
            # Comparing token by token is meaningless once tied tokens
            # have swapped places, so compare the distributions.
            difference = max(
                (
                    abs(left - right)
                    for left, right in zip(kept_expected, kept_received)
                ),
                default=0.0,
            )
        else:
            difference = max(
                abs(left - right) for left, right in zip(expected, received)
            )
        largest_difference = max(largest_difference, difference)
        if not (over_kept or under_kept):
            # Skipped where tied tokens swapped places, because a
            # token-by-token comparison would measure the swap.
            largest_float32_difference = max(
                largest_float32_difference,
                max(
                    abs(left - right)
                    for left, right in zip(single, received)
                ),
            )
        largest_reference_rounding = max(
            largest_reference_rounding,
            max(abs(left - right) for left, right in zip(single, expected)),
        )

        if difference > PROBABILITY_TOLERANCE or support_differs:
            mismatch_count += 1
            if len(mismatches) < REPORTED_MISMATCH_LIMIT:
                mismatches.append(
                    f"  case {index}: size={len(scores)} "
                    f"temperature={temperature} top_k={top_k} "
                    f"top_p={top_p}\n"
                    f"    largest probability difference: {difference:.3e}\n"
                    f"    wrongly kept: {len(over_kept)}, "
                    f"wrongly dropped: {len(under_kept)}"
                )

    print("GPT-2 sampling filter parity")
    print(f"  Transformers: {transformers.__version__}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  filter configurations compared: {len(cases):,}")
    print(f"  probabilities compared: {compared_values:,}")
    print(f"  kept-token-set mismatches: {support_mismatch_count}")
    print(f"  tokens too small to store as float32: {underflowed:,}")
    print(
        "  configurations where tied scores make the boundary "
        f"ambiguous: {tie_ambiguous}"
    )
    print(
        "  largest probability difference from the float64 reference: "
        f"{largest_difference:.6e}"
    )
    print(
        "  largest probability difference from the float32 reference: "
        f"{largest_float32_difference:.6e}"
    )
    print(
        "  largest float32-versus-float64 reference difference: "
        f"{largest_reference_rounding:.6e}"
    )
    print(f"  mismatches: {mismatch_count}")

    if mismatch_count != 0:
        raise AssertionError(
            "GPT2CPP and Hugging Face sampling filters differ:\n"
            + "\n".join(mismatches)
        )


if __name__ == "__main__":
    main()
