from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import prod
from pathlib import Path
import sys
from typing import Any, Union

import numpy as np

from checkpoint_writer import INT8, ModelConfig, TensorRecord, write_checkpoint


DEFAULT_MODEL = "openai-community/gpt2"
DEFAULT_OUTPUT = Path("models/gpt2-small-fp32.bin")

# Weight names quantized under --quantize, matched by suffix so every
# transformer layer's copy is caught. The tied embedding / LM head
# (transformer.wte.weight) is quantized separately, under
# --quantize-tied-embedding, since it is not one of these four and its
# per-channel axis differs from theirs — see _QUANTIZATION_AXIS below.
_QUANTIZED_TRANSFORMER_WEIGHT_SUFFIXES = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)

_TIED_EMBEDDING_NAME = "transformer.wte.weight"

# Every quantized transformer weight is rank 2, laid out [in, out], so
# its output channels are its columns (axis 1) — matching how
# gpt2::linear treats the weight in the C++ inference engine. The tied
# embedding is different: linear()/matmul() never see it. It is read
# row-wise for embedding lookup and, for the tied LM-head projection,
# each row is dotted whole against the hidden state — so its output
# channels (one per vocabulary word) are its rows (axis 0), not its
# columns. This per-tensor axis is exactly why quantization is applied
# tensor by tensor rather than by one fixed rule.
_TRANSFORMER_WEIGHT_AXIS = 1
_TIED_EMBEDDING_AXIS = 0


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


def _tensor_to_numpy(tensor: Any) -> np.ndarray:
    array = tensor.detach().cpu().float().contiguous().numpy()
    # A real torch tensor's .numpy() is already shaped like the
    # tensor; the fake tensor this project's own tests use instead
    # returns a flat buffer, so an explicit reshape is what makes this
    # helper's contract -- "shaped like `tensor`" -- hold for both
    # rather than only for the case that happened not to need it.
    return np.asarray(array, dtype=np.float32).reshape(tuple(tensor.shape))


def _encode_fp32_tensor(tensor: Any) -> memoryview:
    if sys.byteorder != "little":
        raise RuntimeError("GPT2CPP checkpoint export requires a little-endian host")

    array = _tensor_to_numpy(tensor)
    try:
        return memoryview(array).cast("B")
    except (TypeError, ValueError) as error:
        raise ValueError("model tensor must have contiguous FP32 storage") from error


def quantize_per_channel(
    weights: np.ndarray,
    axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric, per-channel int8 quantization along `axis`.

    For each index i along `axis`, the channel's scale is
    s_i = max(|W_i|) / 127 and its values quantize as
    Q_i = round(W_i / s_i), Q_i in [-127, 127]; W_i is recovered as
    approximately s_i * Q_i. A channel that is all zero gets scale 0
    and quantizes to all zero, rather than dividing by zero.

    Returns (quantized, scale): quantized has weights' shape and dtype
    int8; scale is a 1-D float32 array of length weights.shape[axis].
    """
    if weights.ndim < 1 or not (0 <= axis < weights.ndim):
        raise ValueError(
            f"quantization axis {axis} is out of range for a "
            f"{weights.ndim}-dimensional tensor"
        )

    # The division and round happen in float64 so that a channel's own
    # maximum-magnitude element reliably quantizes to exactly ±127
    # rather than landing one step short or over from float32 rounding
    # in the division itself.
    values = weights.astype(np.float64)
    reduce_axes = tuple(axis_index for axis_index in range(values.ndim) if axis_index != axis)
    max_abs = np.max(np.abs(values), axis=reduce_axes)

    scale = max_abs / 127.0
    divisor = np.where(scale > 0.0, scale, 1.0)

    broadcast_shape = [1] * values.ndim
    broadcast_shape[axis] = values.shape[axis]
    quantized = np.round(values / divisor.reshape(broadcast_shape))

    # A defensive clip, not just a correctness belt-and-braces: casting
    # a float outside [-128, 127] straight to int8 wraps around in
    # NumPy rather than raising, so an unclipped 128 would silently
    # become -128 -- exactly the value this scheme cannot represent.
    quantized = np.clip(quantized, -127, 127).astype(np.int8)

    return quantized, scale.astype(np.float32)


def _quantization_axis(
    name: str,
    *,
    quantize_transformer_weights: bool,
    quantize_tied_embedding: bool,
) -> int | None:
    if quantize_tied_embedding and name == _TIED_EMBEDDING_NAME:
        return _TIED_EMBEDDING_AXIS

    if quantize_transformer_weights and any(
        name.endswith(suffix)
        for suffix in _QUANTIZED_TRANSFORMER_WEIGHT_SUFFIXES
    ):
        return _TRANSFORMER_WEIGHT_AXIS

    return None


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
    quantize_transformer_weights: bool = False,
    quantize_tied_embedding: bool = False,
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

        axis = _quantization_axis(
            spec.name,
            quantize_transformer_weights=quantize_transformer_weights,
            quantize_tied_embedding=quantize_tied_embedding,
        )
        if axis is None:
            records.append(
                TensorRecord(
                    name=spec.name,
                    shape=spec.shape,
                    payload=encode_tensor(tensor),
                )
            )
            continue

        if sys.byteorder != "little":
            raise RuntimeError(
                "GPT2CPP checkpoint export requires a little-endian host"
            )

        weights = _tensor_to_numpy(tensor)
        quantized, scale = quantize_per_channel(weights, axis)
        if scale.shape != (spec.shape[axis],):
            raise ValueError(
                f"tensor {spec.name} quantization scale has shape "
                f"{scale.shape}, expected ({spec.shape[axis]},)"
            )

        records.append(
            TensorRecord(
                name=spec.name,
                shape=spec.shape,
                payload=quantized.tobytes(),
                dtype=INT8,
            )
        )
        records.append(
            TensorRecord(
                name=f"{spec.name}.quant_scale",
                shape=scale.shape,
                payload=scale.tobytes(),
            )
        )

    return config, records


def export_model(
    model: Any,
    output_path: str | Path,
    *,
    encode_tensor: TensorEncoder = _encode_fp32_tensor,
    tensors_equal: TensorEquality = _torch_tensors_equal,
    quantize_transformer_weights: bool = False,
    quantize_tied_embedding: bool = False,
) -> ExportSummary:
    config, records = collect_tensor_records(
        model,
        encode_tensor=encode_tensor,
        tensors_equal=tensors_equal,
        quantize_transformer_weights=quantize_transformer_weights,
        quantize_tied_embedding=quantize_tied_embedding,
    )
    write_checkpoint(output_path, config, records)

    output_path = Path(output_path)
    # A per-channel scale is quantization metadata, not a model weight,
    # so it is excluded here to keep parameter_count directly
    # comparable across an FP32 export and its quantized counterparts,
    # which otherwise represent exactly the same trained model.
    parameter_records = (
        record
        for record in records
        if not record.name.endswith(".quant_scale")
    )
    return ExportSummary(
        tensor_count=len(records),
        parameter_count=sum(prod(record.shape) for record in parameter_records),
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
    parser.add_argument(
        "--quantize",
        action="store_true",
        help=(
            "quantize c_attn, c_proj and c_fc weights to symmetric, "
            "per-channel int8"
        ),
    )
    parser.add_argument(
        "--quantize-tied-embedding",
        action="store_true",
        help=(
            "also quantize the tied token embedding / LM-head weight "
            "(transformer.wte.weight); has no effect without --quantize"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model = load_model(args.model, args.local_files_only)
    summary = export_model(
        model,
        args.output,
        quantize_transformer_weights=args.quantize,
        quantize_tied_embedding=args.quantize and args.quantize_tied_embedding,
    )

    print(f"Exported model: {args.model}")
    print(f"Checkpoint: {args.output}")
    print(f"Quantized transformer weights: {args.quantize}")
    print(
        "Quantized tied embedding: "
        f"{args.quantize and args.quantize_tied_embedding}"
    )
    print(f"Tensors: {summary.tensor_count:,}")
    print(f"Unique parameters: {summary.parameter_count:,}")
    print(f"File size: {summary.file_size:,} bytes")


if __name__ == "__main__":
    main()
