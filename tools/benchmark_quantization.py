#!/usr/bin/env python3
"""Runs the three-config quantization benchmark end to end.

For each of the FP32 baseline, int8-transformer-weights and
int8-everything-including-wte configurations (see docs/quantization.md),
this records:

- checkpoint size (file size) and SHA-256
- tokens per second, cached and uncached (gpt2_generation_benchmark)
- peak resident set size over that run
- mean and maximum absolute logit error against the FP32 baseline,
  top-1 and top-5 agreement with it, and perplexity on its own --
  all measured over one real, coherent piece of English text
  (benchmarks/quantization_eval_corpus.txt), not a handful of
  synthetic or hand-picked token IDs, via
  gpt2_quantization_accuracy_runner

into one JSON report, augmented with the same CPU/OS provenance
tools/record_baseline.py adds to the plain generation benchmark.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from record_baseline import cpu_model, os_version  # noqa: E402


CONFIGS = ("fp32", "int8_transformer", "int8_full")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the three-config INT8-quantization benchmark "
            "(checkpoint size, speed, memory, and accuracy vs FP32) "
            "and record one JSON report."
        )
    )
    parser.add_argument("--accuracy-runner", type=Path, required=True)
    parser.add_argument("--generation-benchmark", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--merges", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--fp32-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--int8-transformer-checkpoint", type=Path, required=True
    )
    parser.add_argument("--int8-full-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int)
    parser.add_argument("--new-tokens", type=int)
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--trials", type=int)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        while chunk := checkpoint.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run_command(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{command[0]} exited {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    return completed.stdout


def run_accuracy(
    runner: Path,
    checkpoint: Path,
    vocab: Path,
    merges: Path,
    text: Path,
    output_path: Path,
) -> tuple[np.ndarray, list[int]]:
    stdout = run_command(
        [
            str(runner),
            "--checkpoint",
            str(checkpoint),
            "--vocab",
            str(vocab),
            "--merges",
            str(merges),
            "--text",
            str(text),
            "--output",
            str(output_path),
        ]
    )
    lines = stdout.splitlines()
    if len(lines) != 2:
        raise RuntimeError(
            "accuracy runner produced unexpected output: "
            f"{stdout!r}"
        )

    rows, columns = (int(field) for field in lines[0].split())
    token_ids = [int(field) for field in lines[1].split()]
    if len(token_ids) != rows:
        raise RuntimeError(
            "accuracy runner's token count does not match its logits' "
            "row count"
        )

    logits = np.fromfile(output_path, dtype=np.float32)
    if logits.size != rows * columns:
        raise RuntimeError(
            "accuracy runner output size does not match its reported "
            "shape"
        )
    return logits.reshape(rows, columns), token_ids


def perplexity(logits: np.ndarray, token_ids: list[int]) -> float:
    # Standard causal-LM perplexity: position i predicts token i + 1,
    # so the final position has no target and is excluded.
    logits64 = logits.astype(np.float64)
    log_probabilities = logits64 - np.logaddexp.reduce(
        logits64, axis=1, keepdims=True
    )
    targets = np.asarray(token_ids[1:], dtype=np.int64)
    predicted = log_probabilities[:-1]
    chosen = predicted[np.arange(len(targets)), targets]
    mean_negative_log_likelihood = -float(np.mean(chosen))
    return float(np.exp(mean_negative_log_likelihood))


def run_speed(
    runner: Path,
    checkpoint: Path,
    args: argparse.Namespace,
    json_path: Path,
) -> dict:
    command = [
        str(runner),
        "--checkpoint",
        str(checkpoint),
        "--json",
        str(json_path),
    ]
    if args.prompt_tokens is not None:
        command += ["--prompt-tokens", str(args.prompt_tokens)]
    if args.new_tokens is not None:
        command += ["--new-tokens", str(args.new_tokens)]
    if args.warmups is not None:
        command += ["--warmups", str(args.warmups)]
    if args.trials is not None:
        command += ["--trials", str(args.trials)]

    run_command(command)
    return json.loads(json_path.read_text(encoding="utf-8"))


def compare_to_baseline(
    logits: np.ndarray,
    baseline_logits: np.ndarray,
) -> dict:
    if logits.shape != baseline_logits.shape:
        raise RuntimeError(
            "logit shape mismatch: "
            f"{logits.shape} vs baseline {baseline_logits.shape}"
        )

    differences = np.abs(
        logits.astype(np.float64) - baseline_logits.astype(np.float64)
    )
    baseline_top1 = np.argmax(baseline_logits, axis=1)
    top1 = np.argmax(logits, axis=1)
    top1_agreement = float(np.mean(top1 == baseline_top1))

    # top-5 agreement: does this config's top-5 still contain the
    # position the FP32 baseline would have chosen? (Not "do the two
    # top-5 sets match exactly" -- with real quantization noise, the
    # boundary of a top-5 set is inherently unstable, so exact-set
    # equality would be a far harsher, less meaningful metric than
    # whether the reference choice remains a live candidate.)
    top5 = np.argsort(-logits, axis=1)[:, :5]
    top5_agreement = float(
        np.mean(
            [
                baseline_top1[position] in top5[position]
                for position in range(top5.shape[0])
            ]
        )
    )

    return {
        "mean_absolute_logit_error": float(differences.mean()),
        "max_absolute_logit_error": float(differences.max()),
        "top1_agreement": top1_agreement,
        "top5_agreement": top5_agreement,
    }


def main() -> None:
    args = parse_args()

    # Keep --help and argument-validation usable with the system Python.
    # NumPy is needed only after the benchmark actually starts, and the
    # project CMake smoke test should not require the optional data-science
    # dependency merely to print usage text.
    global np
    try:
        import numpy as np
    except ModuleNotFoundError as exception:
        raise RuntimeError(
            "NumPy is required to run the quantization benchmark; "
            "use the project .venv or install numpy"
        ) from exception

    checkpoints = {
        "fp32": args.fp32_checkpoint,
        "int8_transformer": args.int8_transformer_checkpoint,
        "int8_full": args.int8_full_checkpoint,
    }
    for name, path in checkpoints.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} checkpoint not found: {path}")
    if not args.accuracy_runner.is_file():
        raise FileNotFoundError(
            f"accuracy runner not found: {args.accuracy_runner}"
        )
    if not args.generation_benchmark.is_file():
        raise FileNotFoundError(
            f"generation benchmark not found: {args.generation_benchmark}"
        )

    configs: dict[str, dict] = {}
    logits_by_config: dict[str, np.ndarray] = {}
    token_ids: list[int] | None = None

    with tempfile.TemporaryDirectory() as directory:
        workdir = Path(directory)

        for name in CONFIGS:
            checkpoint = checkpoints[name]
            print(f"=== {name} ===")

            print("  accuracy pass...")
            logits, ids = run_accuracy(
                args.accuracy_runner,
                checkpoint,
                args.vocab,
                args.merges,
                args.text,
                workdir / f"{name}-logits.bin",
            )
            if token_ids is None:
                token_ids = ids
            elif ids != token_ids:
                raise RuntimeError(
                    f"{name} encoded the evaluation text to a different "
                    "token sequence than the FP32 baseline"
                )
            logits_by_config[name] = logits

            print("  speed pass...")
            speed = run_speed(
                args.generation_benchmark,
                checkpoint,
                args,
                workdir / f"{name}-speed.json",
            )

            configs[name] = {
                "checkpoint_path": str(checkpoint),
                "checkpoint_bytes": checkpoint.stat().st_size,
                "checkpoint_sha256": sha256(checkpoint),
                "tokens_per_second": {
                    "cached": speed["results"]["cached_tokens_per_second"],
                    "uncached": (
                        speed["results"]["uncached_tokens_per_second"]
                    ),
                },
                "peak_resident_set_bytes": (
                    speed["results"]["peak_resident_set_bytes"]
                ),
                "perplexity": perplexity(logits_by_config[name], token_ids),
            }

    assert token_ids is not None  # every config ran or main() raised above
    baseline_logits = logits_by_config["fp32"]
    for name in ("int8_transformer", "int8_full"):
        configs[name]["vs_fp32"] = compare_to_baseline(
            logits_by_config[name], baseline_logits
        )

    report = {
        "schema_version": 1,
        "provenance": {
            "cpu_model": cpu_model(),
            "os_version": os_version(),
            "recorded_at": datetime.datetime.now(
                datetime.timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "evaluation_text": str(args.text),
        "evaluation_token_count": len(token_ids),
        "configs": configs,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    print()
    print(f"evaluation text: {args.text} ({len(token_ids)} tokens)")
    for name in CONFIGS:
        entry = configs[name]
        print(f"\n{name}:")
        print(f"  checkpoint bytes: {entry['checkpoint_bytes']:,}")
        print(
            "  tokens/s: "
            f"cached {entry['tokens_per_second']['cached']:.2f}, "
            f"uncached {entry['tokens_per_second']['uncached']:.2f}"
        )
        peak_rss = entry["peak_resident_set_bytes"]
        if peak_rss is not None:
            print(f"  peak RSS: {peak_rss:,} bytes")
        print(f"  perplexity: {entry['perplexity']:.4f}")
        if "vs_fp32" in entry:
            comparison = entry["vs_fp32"]
            print(
                "  vs FP32: mean |error| "
                f"{comparison['mean_absolute_logit_error']:.4e}, "
                f"max |error| {comparison['max_absolute_logit_error']:.4e}"
            )
            print(
                "  vs FP32: top-1 agreement "
                f"{comparison['top1_agreement'] * 100:.2f}%, "
                f"top-5 agreement {comparison['top5_agreement'] * 100:.2f}%"
            )

    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exception:  # noqa: BLE001
        print(f"benchmark_quantization failed: {exception}", file=sys.stderr)
        sys.exit(1)
