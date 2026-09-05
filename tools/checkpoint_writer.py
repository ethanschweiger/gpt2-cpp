from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Sequence


MAGIC = b"GPT2CPP\0"
VERSION = 1
HEADER_SIZE = 64
ENDIAN_MARKER = 0x01020304
FLOAT32 = 1
INT8 = 2

GLOBAL_HEADER = struct.Struct("<8s14I")
TENSOR_HEADER = struct.Struct("<IIIIQQ")

_BYTES_PER_ELEMENT = {FLOAT32: 4, INT8: 1}
_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context_length: int
    embedding_size: int
    head_count: int
    layer_count: int


@dataclass(frozen=True)
class TensorRecord:
    name: str
    shape: tuple[int, ...]
    payload: bytes | memoryview
    dtype: int = FLOAT32


@dataclass(frozen=True)
class _PreparedTensor:
    name: bytes
    shape: tuple[int, ...]
    payload: memoryview
    element_count: int
    dtype: int


def _require_positive_uint32(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")

    if value <= 0 or value > _UINT32_MAX:
        raise ValueError(f"{field_name} must fit in a positive uint32")


def _validate_model_config(config: ModelConfig) -> None:
    fields = (
        (config.vocab_size, "vocabulary size"),
        (config.context_length, "context length"),
        (config.embedding_size, "embedding size"),
        (config.head_count, "head count"),
        (config.layer_count, "layer count"),
    )

    for value, field_name in fields:
        _require_positive_uint32(value, field_name)

    if config.embedding_size % config.head_count != 0:
        raise ValueError("embedding size must be divisible by head count")


def _checked_element_count(shape: tuple[int, ...], bytes_per_element: int) -> int:
    if not shape:
        raise ValueError("tensor shape must have at least one dimension")

    if len(shape) > _UINT32_MAX:
        raise ValueError("tensor rank does not fit in uint32")

    element_count = 1

    for dimension in shape:
        if not isinstance(dimension, int) or isinstance(dimension, bool):
            raise TypeError("tensor dimensions must be integers")

        if dimension <= 0 or dimension > _UINT64_MAX:
            raise ValueError("tensor dimensions must fit in a positive uint64")

        if element_count > _UINT64_MAX // dimension:
            raise OverflowError("tensor element count overflows uint64")

        element_count *= dimension

    if element_count > _UINT64_MAX // bytes_per_element:
        raise OverflowError("tensor payload size overflows uint64")

    return element_count


def _prepare_tensors(tensors: Sequence[TensorRecord]) -> list[_PreparedTensor]:
    if len(tensors) > _UINT32_MAX:
        raise ValueError("tensor count does not fit in uint32")

    prepared = []
    names = set()

    for tensor in tensors:
        if not tensor.name:
            raise ValueError("tensor names must not be empty")

        if tensor.name in names:
            raise ValueError(f"duplicate tensor name: {tensor.name}")

        if tensor.dtype not in _BYTES_PER_ELEMENT:
            raise ValueError(
                f"tensor {tensor.name} has an unsupported dtype: {tensor.dtype}"
            )

        names.add(tensor.name)
        name = tensor.name.encode("utf-8")

        if len(name) > _UINT32_MAX:
            raise ValueError("tensor name length does not fit in uint32")

        bytes_per_element = _BYTES_PER_ELEMENT[tensor.dtype]
        shape = tuple(tensor.shape)
        element_count = _checked_element_count(shape, bytes_per_element)

        try:
            payload = memoryview(tensor.payload).cast("B")
        except (TypeError, ValueError) as error:
            raise ValueError("tensor payload must be contiguous bytes") from error

        expected_size = element_count * bytes_per_element
        if payload.nbytes != expected_size:
            raise ValueError(
                f"tensor {tensor.name} requires {expected_size} payload bytes, "
                f"received {payload.nbytes}"
            )

        if tensor.dtype == INT8:
            for byte in payload:
                if byte == 0x80:
                    raise ValueError(
                        f"tensor {tensor.name} contains a value with no "
                        "symmetric-quantization counterpart: -128"
                    )

        prepared.append(
            _PreparedTensor(
                name=name,
                shape=shape,
                payload=payload,
                element_count=element_count,
                dtype=tensor.dtype,
            )
        )

    return prepared


def write_checkpoint(
    output_path: str | Path,
    config: ModelConfig,
    tensors: Sequence[TensorRecord],
) -> None:
    _validate_model_config(config)
    prepared_tensors = _prepare_tensors(tensors)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as output:
        output.write(
            GLOBAL_HEADER.pack(
                MAGIC,
                VERSION,
                HEADER_SIZE,
                ENDIAN_MARKER,
                0,
                len(prepared_tensors),
                config.vocab_size,
                config.context_length,
                config.embedding_size,
                config.head_count,
                config.layer_count,
                0,
                0,
                0,
                0,
            )
        )

        for tensor in prepared_tensors:
            output.write(
                TENSOR_HEADER.pack(
                    len(tensor.name),
                    tensor.dtype,
                    len(tensor.shape),
                    0,
                    tensor.element_count,
                    tensor.payload.nbytes,
                )
            )
            output.write(
                struct.pack(f"<{len(tensor.shape)}Q", *tensor.shape)
            )
            output.write(tensor.name)
            output.write(tensor.payload)
