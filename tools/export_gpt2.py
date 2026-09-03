from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import prod
from pathlib import Path
import sys
from typing import Any, Union

from checkpoint_writer import ModelConfig, TensorRecord, write_checkpoint


DEFAULT_MODEL = "openai-community/gpt2"
DEFAULT_OUTPUT = Path("models/gpt2-small-fp32.bin")


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class ExportSummary:
    tensor_count: int
    parameter_count: int
    file_size: int


TensorEncoder = Callable[[Any], Union[bytes, memoryview]]
TensorEquality = Callable[[Any, Any], bool]


def _require_positive_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")

    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero")

    return value


def _require_config_value(
    config: Any,
    attribute: str,
    expected: Any,
) -> None:
    actual = getattr(config, attribute, expected)
    if actual != expected:
        raise ValueError(
            f"unsupported GPT-2 configuration: {attribute} must be "
            f"{expected!r}, received {actual!r}"
        )


def model_config_from_hf(hf_config: Any) -> ModelConfig:
    if getattr(hf_config, "model_type", None) != "gpt2":
        raise ValueError("only Hugging Face GPT-2 models are supported")

    vocab_size = _require_positive_int(
        getattr(hf_config, "vocab_size", None),
        "vocabulary size",
    )
    context_length = _require_positive_int(
        getattr(hf_config, "n_positions", None),
        "context length",
    )
    embedding_size = _require_positive_int(
        getattr(hf_config, "n_embd", None),
        "embedding size",
    )
    head_count = _require_positive_int(
        getattr(hf_config, "n_head", None),
        "attention-head count",
    )
    layer_count = _require_positive_int(
        getattr(hf_config, "n_layer", None),
        "transformer-layer count",
    )

    if embedding_size % head_count != 0:
        raise ValueError("embedding size must be divisible by attention-head count")

    if hasattr(hf_config, "n_ctx") and hf_config.n_ctx != context_length:
        raise ValueError("n_ctx must match n_positions")

    inner_size = getattr(hf_config, "n_inner", None)
    if inner_size not in (None, 4 * embedding_size):
        raise ValueError("only the standard GPT-2 MLP width of 4 * n_embd is supported")

    _require_config_value(hf_config, "add_cross_attention", False)
    _require_config_value(hf_config, "tie_word_embeddings", True)
    _require_config_value(hf_config, "activation_function", "gelu_new")
    _require_config_value(hf_config, "layer_norm_epsilon", 1e-5)
    _require_config_value(hf_config, "scale_attn_weights", True)
    _require_config_value(
        hf_config,
        "scale_attn_by_inverse_layer_idx",
        False,
    )
    _require_config_value(hf_config, "reorder_and_upcast_attn", False)

    return ModelConfig(
        vocab_size=vocab_size,
        context_length=context_length,
        embedding_size=embedding_size,
        head_count=head_count,
        layer_count=layer_count,
    )


def expected_tensor_specs(config: ModelConfig) -> list[TensorSpec]:
    vocab_size = _require_positive_int(config.vocab_size, "vocabulary size")
    context_length = _require_positive_int(
        config.context_length,
        "context length",
    )
    embedding_size = _require_positive_int(
        config.embedding_size,
        "embedding size",
    )
    layer_count = _require_positive_int(
        config.layer_count,
        "transformer-layer count",
    )
    head_count = _require_positive_int(
        config.head_count,
        "attention-head count",
    )

    if embedding_size % head_count != 0:
        raise ValueError("embedding size must be divisible by attention-head count")

    specs = [
        TensorSpec(
            "transformer.wte.weight",
            (vocab_size, embedding_size),
        ),
        TensorSpec(
            "transformer.wpe.weight",
            (context_length, embedding_size),
        ),
    ]

    for layer in range(layer_count):
        prefix = f"transformer.h.{layer}"
        specs.extend(
            (
                TensorSpec(f"{prefix}.ln_1.weight", (embedding_size,)),
                TensorSpec(f"{prefix}.ln_1.bias", (embedding_size,)),
                TensorSpec(
                    f"{prefix}.attn.c_attn.weight",
                    (embedding_size, 3 * embedding_size),
                ),
                TensorSpec(
                    f"{prefix}.attn.c_attn.bias",
                    (3 * embedding_size,),
                ),
                TensorSpec(
                    f"{prefix}.attn.c_proj.weight",
                    (embedding_size, embedding_size),
                ),
                TensorSpec(
                    f"{prefix}.attn.c_proj.bias",
                    (embedding_size,),
                ),
                TensorSpec(f"{prefix}.ln_2.weight", (embedding_size,)),
                TensorSpec(f"{prefix}.ln_2.bias", (embedding_size,)),
                TensorSpec(
                    f"{prefix}.mlp.c_fc.weight",
                    (embedding_size, 4 * embedding_size),
                ),
                TensorSpec(
                    f"{prefix}.mlp.c_fc.bias",
                    (4 * embedding_size,),
                ),
                TensorSpec(
                    f"{prefix}.mlp.c_proj.weight",
                    (4 * embedding_size, embedding_size),
                ),
                TensorSpec(
                    f"{prefix}.mlp.c_proj.bias",
                    (embedding_size,),
                ),
            )
        )

    specs.extend(
        (
            TensorSpec("transformer.ln_f.weight", (embedding_size,)),
            TensorSpec("transformer.ln_f.bias", (embedding_size,)),
        )
    )

    return specs


def _encode_fp32_tensor(tensor: Any) -> memoryview:
    if sys.byteorder != "little":
        raise RuntimeError("GPT2CPP checkpoint export requires a little-endian host")

    array = tensor.detach().cpu().float().contiguous().numpy()
    try:
        return memoryview(array).cast("B")
    except (TypeError, ValueError) as error:
        raise ValueError("model tensor must have contiguous FP32 storage") from error


def _torch_tensors_equal(left: Any, right: Any) -> bool:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required; install requirements-export.txt"
        ) from error

    return bool(torch.equal(left.detach().cpu(), right.detach().cpu()))


def _ignored_transformer_buffers(layer_count: int) -> set[str]:
    names = set()
    for layer in range(layer_count):
        names.add(f"h.{layer}.attn.bias")
        names.add(f"h.{layer}.attn.masked_bias")
    return names


def collect_tensor_records(
    model: Any,
    *,
    encode_tensor: TensorEncoder = _encode_fp32_tensor,
    tensors_equal: TensorEquality = _torch_tensors_equal,
) -> tuple[ModelConfig, list[TensorRecord]]:
    config = model_config_from_hf(model.config)
    specs = expected_tensor_specs(config)

    transformer = getattr(model, "transformer", None)
    if transformer is None or not callable(getattr(transformer, "state_dict", None)):
        raise ValueError("model does not expose a GPT-2 transformer state dictionary")

    state = transformer.state_dict()
    if not isinstance(state, Mapping):
        raise TypeError("transformer state_dict() must return a mapping")

    expected_state_names = {
        spec.name.removeprefix("transformer.") for spec in specs
    }
    ignored_state_names = _ignored_transformer_buffers(config.layer_count)
    missing = sorted(expected_state_names.difference(state))
    if missing:
        raise ValueError(f"model is missing tensor: transformer.{missing[0]}")

    unexpected = sorted(
        name
        for name in state
        if name not in expected_state_names
        and name not in ignored_state_names
    )
    if unexpected:
        raise ValueError(f"model contains unsupported tensor: transformer.{unexpected[0]}")

    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if input_embeddings is None or output_embeddings is None:
        raise ValueError("model must expose input and output embedding weights")

    if not tensors_equal(input_embeddings.weight, output_embeddings.weight):
        raise ValueError("GPT-2 output weights must match the token embedding weights")

    records = []
    for spec in specs:
        state_name = spec.name.removeprefix("transformer.")
        tensor = state[state_name]
        actual_shape = tuple(tensor.shape)
        if actual_shape != spec.shape:
            raise ValueError(
                f"tensor {spec.name} has shape {actual_shape}, "
                f"expected {spec.shape}"
            )

        records.append(
            TensorRecord(
                name=spec.name,
                shape=spec.shape,
                payload=encode_tensor(tensor),
            )
        )

    return config, records


def export_model(
    model: Any,
    output_path: str | Path,
    *,
    encode_tensor: TensorEncoder = _encode_fp32_tensor,
    tensors_equal: TensorEquality = _torch_tensors_equal,
) -> ExportSummary:
    config, records = collect_tensor_records(
        model,
        encode_tensor=encode_tensor,
        tensors_equal=tensors_equal,
    )
    write_checkpoint(output_path, config, records)

    output_path = Path(output_path)
    return ExportSummary(
        tensor_count=len(records),
        parameter_count=sum(prod(record.shape) for record in records),
        file_size=output_path.stat().st_size,
    )


def load_model(model_name: str, local_files_only: bool = False) -> Any:
    try:
        from transformers import GPT2LMHeadModel
    except ImportError as error:
        raise RuntimeError(
            "Hugging Face Transformers is required; "
            "install requirements-export.txt"
        ) from error

    model = GPT2LMHeadModel.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    model.eval()
    model.cpu()
    model.float()
    return model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Hugging Face GPT-2 weights to GPT2CPP format."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Hugging Face model name or local directory (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"checkpoint output path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="do not download model files",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model = load_model(args.model, args.local_files_only)
    summary = export_model(model, args.output)

    print(f"Exported model: {args.model}")
    print(f"Checkpoint: {args.output}")
    print(f"Tensors: {summary.tensor_count:,}")
    print(f"Unique parameters: {summary.parameter_count:,}")
    print(f"File size: {summary.file_size:,} bytes")


if __name__ == "__main__":
    main()
