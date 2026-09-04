from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile


EXPECTED_CHECKPOINT_SHA256 = (
    "c506385a3b29873ef9148a8a5b672b66711a01c2180865ae46449f74cac0d6dc"
)
EXPECTED_VOCABULARY_SHA256 = (
    "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783"
)
EXPECTED_MERGES_SHA256 = (
    "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5"
)
EXPECTED_CONTEXT_LENGTH = 1024
END_OF_TEXT_ID = 50256

# Greedy generation re-runs the whole forward pass for every new token,
# so the prompts stay short and the budgets modest.
CASES = (
    ("Hello, world!", 12),
    ("The capital of France is", 12),
    ("1, 2, 3, 4,", 8),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare GPT2CPP greedy generation with Hugging Face."
        )
    )
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--merges", type=Path, required=True)
    parser.add_argument("--hf-reference", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as asset:
        while chunk := asset.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run_runner(
    runner: Path,
    checkpoint: Path,
    vocabulary: Path,
    merges: Path,
) -> tuple[list[str], int]:
    requests = [
        f"{budget} " + prompt.encode("utf-8").hex()
        for prompt, budget in CASES
    ]

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
                str(checkpoint),
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
                "C++ generation runner failed:\n" + completed.stderr.strip()
            )

        answers = answer_path.read_text(encoding="ascii").splitlines()

    return answers, int(completed.stdout.strip())


def load_reference(reference: str, vocabulary: Path, merges: Path):
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    model = GPT2LMHeadModel.from_pretrained(
        reference,
        local_files_only=True,
        attn_implementation="eager",
    )
    model.eval()
    model.cpu()
    model.float()

    with tempfile.TemporaryDirectory() as directory:
        shutil.copyfile(vocabulary, Path(directory) / "vocab.json")
        shutil.copyfile(merges, Path(directory) / "merges.txt")
        tokenizer = GPT2TokenizerFast.from_pretrained(
            directory,
            local_files_only=True,
        )

    return model, tokenizer, torch


def main() -> None:
    args = parse_args()

    if not args.runner.is_file():
        raise FileNotFoundError(f"C++ runner not found: {args.runner}")
    for asset, expected in (
        (args.checkpoint, EXPECTED_CHECKPOINT_SHA256),
        (args.vocabulary, EXPECTED_VOCABULARY_SHA256),
        (args.merges, EXPECTED_MERGES_SHA256),
    ):
        if not asset.is_file():
            raise FileNotFoundError(f"asset not found: {asset}")
        digest = sha256(asset)
        if digest != expected:
            raise AssertionError(
                f"{asset.name} SHA-256 mismatch: "
                f"expected {expected}, received {digest}"
            )

    answers, context_length = run_runner(
        args.runner,
        args.checkpoint,
        args.vocabulary,
        args.merges,
    )
    if len(answers) != len(CASES):
        raise AssertionError(
            f"C++ runner answered {len(answers)} of {len(CASES)} requests"
        )

    model, tokenizer, torch = load_reference(
        args.hf_reference,
        args.vocabulary,
        args.merges,
    )

    mismatches: list[str] = []
    generated_token_count = 0

    print("GPT-2 Small greedy generation parity")
    print(f"  checkpoint SHA-256: {sha256(args.checkpoint)}")
    print(f"  GPT2CPP context length: {context_length:,}")

    for index, (prompt, budget) in enumerate(CASES):
        fields = answers[index].split()
        stop_reason = fields[0]
        cpp_ids = [int(field) for field in fields[1:]]

        prompt_ids = tokenizer.encode(prompt)
        with torch.inference_mode():
            reference = model.generate(
                input_ids=torch.tensor([prompt_ids], dtype=torch.long),
                max_new_tokens=budget,
                do_sample=False,
                num_beams=1,
                use_cache=False,
                eos_token_id=END_OF_TEXT_ID,
                pad_token_id=END_OF_TEXT_ID,
            )
        reference_ids = reference[0].tolist()[len(prompt_ids):]
        generated_token_count += len(cpp_ids)

        cpp_text = tokenizer.decode(prompt_ids + cpp_ids)
        reference_text = tokenizer.decode(prompt_ids + reference_ids)

        print()
        print(f"  prompt: {prompt!r}")
        print(f"    prompt tokens: {len(prompt_ids)}")
        print(f"    new tokens: {len(cpp_ids)} (stopped at {stop_reason})")
        print(f"    GPT2CPP:      {cpp_text!r}")
        print(f"    Hugging Face: {reference_text!r}")

        if cpp_ids != reference_ids:
            first = next(
                (
                    position
                    for position, (left, right) in enumerate(
                        zip(cpp_ids, reference_ids)
                    )
                    if left != right
                ),
                min(len(cpp_ids), len(reference_ids)),
            )
            mismatches.append(
                f"  {prompt!r} diverges at new token {first}\n"
                f"    GPT2CPP:      {cpp_ids}\n"
                f"    Hugging Face: {reference_ids}"
            )

    print()
    print(f"  cases: {len(CASES)}")
    print(f"  generated tokens compared: {generated_token_count}")
    print(f"  mismatches: {len(mismatches)}")

    if context_length != EXPECTED_CONTEXT_LENGTH:
        raise AssertionError(
            "context length mismatch: "
            f"expected {EXPECTED_CONTEXT_LENGTH}, received {context_length}"
        )
    if mismatches:
        raise AssertionError(
            "GPT2CPP and Hugging Face generations differ:\n"
            + "\n".join(mismatches)
        )


if __name__ == "__main__":
    main()
