from __future__ import annotations

from dataclasses import dataclass
from math import prod
import os
from pathlib import Path
import struct
import tempfile
import unittest


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from export_gpt2 import export_model


EXPECTED_GLOBAL_HEADER = struct.Struct("<8s14I")
EXPECTED_TENSOR_HEADER = struct.Struct("<IIIIQQ")

EXPECTED_SCHEMA = (
    ("transformer.wte.weight", (32, 8)),
    ("transformer.wpe.weight", (8, 8)),
    ("transformer.h.0.ln_1.weight", (8,)),
    ("transformer.h.0.ln_1.bias", (8,)),
    ("transformer.h.0.attn.c_attn.weight", (8, 24)),
    ("transformer.h.0.attn.c_attn.bias", (24,)),
    ("transformer.h.0.attn.c_proj.weight", (8, 8)),
    ("transformer.h.0.attn.c_proj.bias", (8,)),
    ("transformer.h.0.ln_2.weight", (8,)),
    ("transformer.h.0.ln_2.bias", (8,)),
    ("transformer.h.0.mlp.c_fc.weight", (8, 32)),
    ("transformer.h.0.mlp.c_fc.bias", (32,)),
    ("transformer.h.0.mlp.c_proj.weight", (32, 8)),
    ("transformer.h.0.mlp.c_proj.bias", (8,)),
    ("transformer.h.1.ln_1.weight", (8,)),
    ("transformer.h.1.ln_1.bias", (8,)),
    ("transformer.h.1.attn.c_attn.weight", (8, 24)),
    ("transformer.h.1.attn.c_attn.bias", (24,)),
    ("transformer.h.1.attn.c_proj.weight", (8, 8)),
    ("transformer.h.1.attn.c_proj.bias", (8,)),
    ("transformer.h.1.ln_2.weight", (8,)),
    ("transformer.h.1.ln_2.bias", (8,)),
    ("transformer.h.1.mlp.c_fc.weight", (8, 32)),
    ("transformer.h.1.mlp.c_fc.bias", (32,)),
    ("transformer.h.1.mlp.c_proj.weight", (32, 8)),
    ("transformer.h.1.mlp.c_proj.bias", (8,)),
    ("transformer.ln_f.weight", (8,)),
    ("transformer.ln_f.bias", (8,)),
)


@dataclass(frozen=True)
class ParsedTensor:
    name: str
    data_type: int
    rank: int
    flags: int
    element_count: int
    payload_size: int
    shape: tuple[int, ...]
    values: tuple[float, ...]


def parse_checkpoint(contents: bytes) -> tuple[tuple, list[ParsedTensor], int]:
    global_header = EXPECTED_GLOBAL_HEADER.unpack_from(contents, 0)
    tensor_count = global_header[5]
    offset = EXPECTED_GLOBAL_HEADER.size
    tensors = []

    for _ in range(tensor_count):
        (
            name_length,
            data_type,
            rank,
            flags,
            element_count,
            payload_size,
        ) = EXPECTED_TENSOR_HEADER.unpack_from(contents, offset)
        offset += EXPECTED_TENSOR_HEADER.size

        dimensions = struct.Struct(f"<{rank}Q")
        shape = dimensions.unpack_from(contents, offset)
        offset += dimensions.size

        name = contents[offset : offset + name_length].decode("utf-8")
        offset += name_length

        payload = contents[offset : offset + payload_size]
        offset += payload_size
        values = struct.unpack(f"<{element_count}f", payload)

        tensors.append(
            ParsedTensor(
                name=name,
                data_type=data_type,
                rank=rank,
                flags=flags,
                element_count=element_count,
                payload_size=payload_size,
                shape=shape,
                values=values,
            )
        )

    return global_header, tensors, offset


class HuggingFaceExportIntegrationTest(unittest.TestCase):
    def test_exports_real_tiny_gpt2_model(self) -> None:
        torch.manual_seed(1234)
        config = GPT2Config(
            vocab_size=32,
            n_positions=8,
            n_ctx=8,
            n_embd=8,
            n_head=2,
            n_layer=2,
            bos_token_id=0,
            eos_token_id=1,
        )
        model = GPT2LMHeadModel(config)
        state = model.transformer.state_dict()
        source_values = {}

        with torch.no_grad():
            for tensor_index, (name, expected_shape) in enumerate(EXPECTED_SCHEMA):
                state_name = name.removeprefix("transformer.")
                tensor = state[state_name]
                self.assertEqual(tuple(tensor.shape), expected_shape)

                values = torch.arange(tensor.numel(), dtype=torch.float32)
                values = values.add(tensor_index * 1000).reshape(expected_shape)
                tensor.copy_(values)
                source_values[name] = tuple(tensor.reshape(-1).tolist())

        input_weight = model.get_input_embeddings().weight
        output_weight = model.get_output_embeddings().weight
        self.assertEqual(input_weight.data_ptr(), output_weight.data_ptr())

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "tiny-hugging-face-gpt2.bin"
            summary = export_model(model, output_path)
            contents = output_path.read_bytes()

        global_header, tensors, final_offset = parse_checkpoint(contents)

        self.assertEqual(
            global_header,
            (
                b"GPT2CPP\0",
                1,
                64,
                0x01020304,
                0,
                28,
                32,
                8,
                8,
                2,
                2,
                0,
                0,
                0,
                0,
            ),
        )
        self.assertEqual(
            [(tensor.name, tensor.shape) for tensor in tensors],
            list(EXPECTED_SCHEMA),
        )

        for tensor in tensors:
            with self.subTest(name=tensor.name):
                self.assertEqual(tensor.data_type, 1)
                self.assertEqual(tensor.rank, len(tensor.shape))
                self.assertEqual(tensor.flags, 0)
                self.assertEqual(tensor.element_count, prod(tensor.shape))
                self.assertEqual(tensor.payload_size, tensor.element_count * 4)
                self.assertEqual(tensor.values, source_values[tensor.name])

        self.assertNotIn("lm_head.weight", (tensor.name for tensor in tensors))
        self.assertEqual(
            next(
                tensor.shape
                for tensor in tensors
                if tensor.name == "transformer.h.0.attn.c_attn.weight"
            ),
            (8, 24),
        )
        self.assertEqual(summary.tensor_count, 28)
        self.assertEqual(summary.parameter_count, 2_080)
        self.assertEqual(summary.file_size, len(contents))
        self.assertEqual(final_offset, len(contents))


if __name__ == "__main__":
    unittest.main()
