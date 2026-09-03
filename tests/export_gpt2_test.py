from __future__ import annotations

from array import array
from dataclasses import dataclass
from math import prod
from pathlib import Path
import struct
import tempfile
import unittest

from checkpoint_writer import ModelConfig
from export_gpt2 import (
    collect_tensor_records,
    expected_tensor_specs,
    export_model,
    model_config_from_hf,
)


EXPECTED_ONE_LAYER_SCHEMA = (
    ("transformer.wte.weight", (8, 4)),
    ("transformer.wpe.weight", (4, 4)),
    ("transformer.h.0.ln_1.weight", (4,)),
    ("transformer.h.0.ln_1.bias", (4,)),
    ("transformer.h.0.attn.c_attn.weight", (4, 12)),
    ("transformer.h.0.attn.c_attn.bias", (12,)),
    ("transformer.h.0.attn.c_proj.weight", (4, 4)),
    ("transformer.h.0.attn.c_proj.bias", (4,)),
    ("transformer.h.0.ln_2.weight", (4,)),
    ("transformer.h.0.ln_2.bias", (4,)),
    ("transformer.h.0.mlp.c_fc.weight", (4, 16)),
    ("transformer.h.0.mlp.c_fc.bias", (16,)),
    ("transformer.h.0.mlp.c_proj.weight", (16, 4)),
    ("transformer.h.0.mlp.c_proj.bias", (4,)),
    ("transformer.ln_f.weight", (4,)),
    ("transformer.ln_f.bias", (4,)),
)


@dataclass
class FakeConfig:
    model_type: str = "gpt2"
    vocab_size: int = 8
    n_positions: int = 4
    n_ctx: int = 4
    n_embd: int = 4
    n_head: int = 2
    n_layer: int = 1
    n_inner: int | None = None
    add_cross_attention: bool = False
    tie_word_embeddings: bool = True
    activation_function: str = "gelu_new"
    layer_norm_epsilon: float = 1e-5
    scale_attn_weights: bool = True
    scale_attn_by_inverse_layer_idx: bool = False
    reorder_and_upcast_attn: bool = False


@dataclass(frozen=True)
class FakeTensor:
    shape: tuple[int, ...]
    values: tuple[float, ...]

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def float(self) -> FakeTensor:
        return self

    def contiguous(self) -> FakeTensor:
        return self

    def numpy(self) -> array:
        return array("f", self.values)


@dataclass(frozen=True)
class FakeEmbedding:
    weight: FakeTensor


class FakeTransformer:
    def __init__(self, state: dict[str, FakeTensor]) -> None:
        self._state = state

    def state_dict(self) -> dict[str, FakeTensor]:
        return self._state


class FakeModel:
    def __init__(
        self,
        config: FakeConfig,
        state: dict[str, FakeTensor],
        output_weight: FakeTensor,
    ) -> None:
        self.config = config
        self.transformer = FakeTransformer(state)
        self._input_embedding = FakeEmbedding(state["wte.weight"])
        self._output_embedding = FakeEmbedding(output_weight)

    def get_input_embeddings(self) -> FakeEmbedding:
        return self._input_embedding

    def get_output_embeddings(self) -> FakeEmbedding:
        return self._output_embedding


def fake_encode(tensor: FakeTensor) -> bytes:
    return struct.pack(f"<{len(tensor.values)}f", *tensor.values)


def fake_equal(left: FakeTensor, right: FakeTensor) -> bool:
    return left == right


def make_model(config: FakeConfig | None = None) -> FakeModel:
    config = config or FakeConfig()

    state = {}
    next_value = 1.0
    for tensor_name, shape in EXPECTED_ONE_LAYER_SCHEMA:
        element_count = 1
        for dimension in shape:
            element_count *= dimension

        values = tuple(next_value + index for index in range(element_count))
        next_value += element_count
        state_name = tensor_name.removeprefix("transformer.")
        state[state_name] = FakeTensor(shape, values)

    return FakeModel(config, state, state["wte.weight"])


class ExportGpt2Test(unittest.TestCase):
    def test_converts_hugging_face_configuration(self) -> None:
        self.assertEqual(
            model_config_from_hf(FakeConfig()),
            ModelConfig(
                vocab_size=8,
                context_length=4,
                embedding_size=4,
                head_count=2,
                layer_count=1,
            ),
        )

    def test_collects_expected_tensors_in_stable_order(self) -> None:
        model = make_model()
        _, records = collect_tensor_records(
            model,
            encode_tensor=fake_encode,
            tensors_equal=fake_equal,
        )

        self.assertEqual(len(records), 16)
        self.assertEqual(
            [(record.name, record.shape) for record in records],
            list(EXPECTED_ONE_LAYER_SCHEMA),
        )
        self.assertNotIn("lm_head.weight", (record.name for record in records))

    def test_default_encoder_produces_fp32_payload(self) -> None:
        model = make_model()
        _, records = collect_tensor_records(
            model,
            tensors_equal=fake_equal,
        )
        embedding = records[0]

        self.assertEqual(embedding.payload.nbytes, 8 * 4 * 4)
        self.assertEqual(
            struct.unpack_from("<3f", embedding.payload),
            (1.0, 2.0, 3.0),
        )

    def test_standard_gpt2_has_expected_unique_parameter_count(self) -> None:
        config = ModelConfig(
            vocab_size=50257,
            context_length=1024,
            embedding_size=768,
            head_count=12,
            layer_count=12,
        )
        specs = expected_tensor_specs(config)
        parameter_count = sum(prod(spec.shape) for spec in specs)

        self.assertEqual(len(specs), 148)
        self.assertEqual(parameter_count, 124_439_808)

    def test_allows_legacy_nonpersistent_attention_buffers(self) -> None:
        model = make_model()
        model.transformer._state["h.0.attn.bias"] = FakeTensor((1,), (1.0,))
        model.transformer._state["h.0.attn.masked_bias"] = FakeTensor(
            (1,),
            (-10000.0,),
        )

        _, records = collect_tensor_records(
            model,
            encode_tensor=fake_encode,
            tensors_equal=fake_equal,
        )
        self.assertEqual(len(records), 16)

    def test_rejects_missing_unexpected_and_wrong_shape_tensors(self) -> None:
        cases = []

        missing = make_model()
        del missing.transformer._state["ln_f.bias"]
        cases.append((missing, "missing tensor"))

        unexpected = make_model()
        unexpected.transformer._state["mystery.weight"] = FakeTensor((1,), (1.0,))
        cases.append((unexpected, "unsupported tensor"))

        invalid_buffer = make_model()
        invalid_buffer.transformer._state["h.9.attn.bias"] = FakeTensor(
            (1,),
            (1.0,),
        )
        cases.append((invalid_buffer, "unsupported tensor"))

        wrong_shape = make_model()
        original = wrong_shape.transformer._state["h.0.ln_1.weight"]
        wrong_shape.transformer._state["h.0.ln_1.weight"] = FakeTensor(
            (2, 2),
            original.values,
        )
        cases.append((wrong_shape, "has shape"))

        for model, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    collect_tensor_records(
                        model,
                        encode_tensor=fake_encode,
                        tensors_equal=fake_equal,
                    )

    def test_rejects_untied_output_weights(self) -> None:
        model = make_model()
        input_weight = model.get_input_embeddings().weight
        different_values = (999.0,) + input_weight.values[1:]
        model._output_embedding = FakeEmbedding(
            FakeTensor(input_weight.shape, different_values)
        )

        with self.assertRaisesRegex(ValueError, "output weights must match"):
            collect_tensor_records(
                model,
                encode_tensor=fake_encode,
                tensors_equal=fake_equal,
            )

    def test_rejects_unsupported_gpt2_configurations(self) -> None:
        cases = (
            ("model_type", "bert"),
            ("n_inner", 12),
            ("add_cross_attention", True),
            ("tie_word_embeddings", False),
            ("activation_function", "relu"),
            ("layer_norm_epsilon", 1e-6),
            ("scale_attn_weights", False),
            ("scale_attn_by_inverse_layer_idx", True),
            ("reorder_and_upcast_attn", True),
        )

        for attribute, value in cases:
            with self.subTest(attribute=attribute):
                config = FakeConfig()
                setattr(config, attribute, value)
                with self.assertRaises(ValueError):
                    model_config_from_hf(config)

    def test_exports_fake_model_through_binary_writer(self) -> None:
        model = make_model()

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "tiny-gpt2.bin"
            summary = export_model(
                model,
                output_path,
                encode_tensor=fake_encode,
                tensors_equal=fake_equal,
            )
            contents = output_path.read_bytes()

        self.assertEqual(summary.tensor_count, 16)
        self.assertEqual(summary.file_size, len(contents))
        self.assertEqual(struct.unpack_from("<I", contents, 24), (16,))
        self.assertGreater(summary.parameter_count, 0)


if __name__ == "__main__":
    unittest.main()
