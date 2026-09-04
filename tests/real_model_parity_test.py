from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile


PROMPT_TOKEN_IDS = (15496, 11, 995, 0)  # "Hello, world!"
EXPECTED_CHECKPOINT_SHA256 = (
    "c506385a3b29873ef9148a8a5b672b66711a01c2180865ae46449f74cac0d6dc"
)
MAX_ABSOLUTE_ERROR_LIMIT = 5.0e-4
ROOT_MEAN_SQUARE_ERROR_LIMIT = 1.0e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare GPT2CPP logits with a matching Hugging Face model."
        )
    )
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hf-reference", required=True)
    return parser.parse_args()


def run_cpp_model(
    runner: Path,
    checkpoint: Path,
    output_path: Path,
) -> tuple[int, int]:
    completed = subprocess.run(
        (
            str(runner),
            str(checkpoint),
            str(output_path),
            *(str(token_id) for token_id in PROMPT_TOKEN_IDS),
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "C++ reference runner failed:\n" + completed.stderr.strip()
        )

    fields = completed.stdout.split()
    if len(fields) != 2:
        raise RuntimeError(
            "C++ reference runner returned an invalid output shape"
        )

    return int(fields[0]), int(fields[1])


def load_hugging_face_logits(reference: str):
    import torch
    import transformers
    from transformers import GPT2LMHeadModel

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

    input_ids = torch.tensor(
        [PROMPT_TOKEN_IDS],
        dtype=torch.long,
    )
    with torch.inference_mode():
        logits = model(input_ids=input_ids, use_cache=False).logits

    return (
        logits[0].cpu().numpy(),
        torch.__version__,
        transformers.__version__,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        while chunk := checkpoint.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()

    if not args.runner.is_file():
        raise FileNotFoundError(f"C++ runner not found: {args.runner}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"exported checkpoint not found: {args.checkpoint}"
        )

    checkpoint_sha256 = sha256(args.checkpoint)
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise AssertionError(
            "checkpoint SHA-256 mismatch: "
            f"expected {EXPECTED_CHECKPOINT_SHA256}, "
            f"received {checkpoint_sha256}"
        )

    with tempfile.TemporaryDirectory() as directory:
        output_path = Path(directory) / "cpp-logits-fp32.bin"
        cpp_shape = run_cpp_model(
            args.runner,
            args.checkpoint,
            output_path,
        )

        import numpy as np

        cpp_logits = np.fromfile(output_path, dtype=np.float32)

    expected_value_count = cpp_shape[0] * cpp_shape[1]
    if cpp_logits.size != expected_value_count:
        raise AssertionError(
            "C++ runner output size does not match its reported shape"
        )
    cpp_logits = cpp_logits.reshape(cpp_shape)

    (
        reference_logits,
        torch_version,
        transformers_version,
    ) = load_hugging_face_logits(args.hf_reference)
    if cpp_logits.shape != reference_logits.shape:
        raise AssertionError(
            "logit shape mismatch: "
            f"C++ {cpp_logits.shape}, Hugging Face {reference_logits.shape}"
        )

    if not np.isfinite(cpp_logits).all():
        raise AssertionError("C++ logits contain a non-finite value")
    if not np.isfinite(reference_logits).all():
        raise AssertionError(
            "Hugging Face logits contain a non-finite value"
        )

    differences = np.abs(
        cpp_logits.astype(np.float64) -
        reference_logits.astype(np.float64)
    )
    maximum_error = float(differences.max())
    mean_error = float(differences.mean())
    root_mean_square_error = float(
        np.sqrt(np.mean(np.square(differences)))
    )

    cpp_predictions = np.argmax(cpp_logits, axis=1)
    reference_predictions = np.argmax(reference_logits, axis=1)
    matching_predictions = int(
        np.count_nonzero(cpp_predictions == reference_predictions)
    )
    position_count = cpp_logits.shape[0]

    worst_flat_index = int(np.argmax(differences))
    worst_position, worst_token = np.unravel_index(
        worst_flat_index,
        differences.shape,
    )

    print("GPT-2 Small numerical parity")
    print(f"  checkpoint SHA-256: {checkpoint_sha256}")
    print(f"  NumPy: {np.__version__}")
    print(f"  PyTorch: {torch_version}")
    print(f"  Transformers: {transformers_version}")
    print(f"  prompt token IDs: {list(PROMPT_TOKEN_IDS)}")
    print(f"  compared logits: {cpp_logits.size:,}")
    print(f"  maximum absolute error: {maximum_error:.9e}")
    print(f"  mean absolute error: {mean_error:.9e}")
    print(
        "  root mean square error: "
        f"{root_mean_square_error:.9e}"
    )
    print(
        "  top-1 agreement: "
        f"{matching_predictions}/{position_count} positions"
    )
    print(
        "  largest-error location: "
        f"position {worst_position}, token {worst_token}"
    )

    if maximum_error > MAX_ABSOLUTE_ERROR_LIMIT:
        raise AssertionError(
            "maximum absolute error exceeds the limit: "
            f"{maximum_error:.9e} > {MAX_ABSOLUTE_ERROR_LIMIT:.9e}"
        )
    if root_mean_square_error > ROOT_MEAN_SQUARE_ERROR_LIMIT:
        raise AssertionError(
            "root mean square error exceeds the limit: "
            f"{root_mean_square_error:.9e} > "
            f"{ROOT_MEAN_SQUARE_ERROR_LIMIT:.9e}"
        )
    if matching_predictions != position_count:
        raise AssertionError(
            "C++ and Hugging Face top-1 predictions do not all match"
        )


if __name__ == "__main__":
    main()
