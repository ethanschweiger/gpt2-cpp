#!/usr/bin/env python3
"""Runs the generation benchmark and records a traceable baseline.

Wraps gpt2_generation_benchmark's own JSON with the three fields that
make a timing claim checkable months later and on a different machine:
the checkpoint's exact identity, and the CPU and OS that produced the
numbers. The benchmark binary stays platform-generic; this script is
the only place that shells out to sysctl / sw_vers / /proc/cpuinfo.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run gpt2_generation_benchmark and record a checkpoint- and "
            "machine-identified baseline."
        )
    )
    parser.add_argument(
        "--runner",
        type=Path,
        required=True,
        help="path to the gpt2_generation_benchmark executable",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="where to write the merged baseline JSON",
    )
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
    return completed.stdout.strip()


def cpu_model() -> str:
    system = platform.system()
    if system == "Darwin":
        try:
            return run_command(
                ["sysctl", "-n", "machdep.cpu.brand_string"]
            )
        except (RuntimeError, OSError):
            return "unknown"

    if system == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
        return "unknown"

    return platform.processor() or "unknown"


def os_version() -> str:
    system = platform.system()
    if system == "Darwin":
        try:
            product = run_command(["sw_vers", "-productName"])
            version = run_command(["sw_vers", "-productVersion"])
            build = run_command(["sw_vers", "-buildVersion"])
            return f"{product} {version} ({build})"
        except (RuntimeError, OSError):
            return platform.platform()

    if system == "Linux":
        os_release = Path("/etc/os-release")
        if os_release.is_file():
            fields = dict(
                line.split("=", 1)
                for line in os_release.read_text().splitlines()
                if "=" in line
            )
            pretty_name = fields.get("PRETTY_NAME", "").strip('"')
            if pretty_name:
                return f"{pretty_name}, kernel {platform.release()}"
        return platform.platform()

    return platform.platform()


def build_command(args: argparse.Namespace, json_path: Path) -> list[str]:
    command = [
        str(args.runner),
        "--checkpoint",
        str(args.checkpoint),
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
    return command


def main() -> None:
    args = parse_args()

    if not args.runner.is_file():
        raise FileNotFoundError(f"benchmark runner not found: {args.runner}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    checkpoint_sha256 = sha256(args.checkpoint)

    with tempfile.TemporaryDirectory() as directory:
        json_path = Path(directory) / "benchmark.json"
        command = build_command(args, json_path)

        print("Running:", " ".join(command))
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"benchmark exited with status {completed.returncode}"
            )

        record = json.loads(json_path.read_text(encoding="utf-8"))

    record["provenance"] = {
        "checkpoint_sha256": checkpoint_sha256,
        "cpu_model": cpu_model(),
        "os_version": os_version(),
        "recorded_at": datetime.datetime.now(
            datetime.timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    results = record["results"]
    print()
    print(f"checkpoint SHA-256: {checkpoint_sha256}")
    print(f"CPU: {record['provenance']['cpu_model']}")
    print(f"OS: {record['provenance']['os_version']}")
    print(
        "cached: "
        f"{results['cached_median_seconds']:.3f} s "
        f"({results['cached_tokens_per_second']:.2f} tokens/s)"
    )
    print(
        "uncached: "
        f"{results['uncached_median_seconds']:.3f} s "
        f"({results['uncached_tokens_per_second']:.2f} tokens/s)"
    )
    print(f"cache speedup: {results['cache_speedup']:.2f}x")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exception:  # noqa: BLE001
        print(f"record_baseline failed: {exception}", file=sys.stderr)
        sys.exit(1)
