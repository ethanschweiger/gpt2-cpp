from __future__ import annotations

from array import array
from dataclasses import dataclass
from math import prod
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from checkpoint_writer import INT8, ModelConfig
from export_gpt2 import (
    collect_tensor_records,
    expected_tensor_specs,
    export_model,
    model_config_from_hf,
    quantize_per_channel,
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


def _reference_quantize_per_channel(
    weights: list[list[float]],
    axis: int,
) -> tuple[list[list[int]], list[float]]:
    """An intentionally naive, un-vectorized restatement of the
    symmetric per-channel scheme, used to check quantize_per_channel
    without sharing any of its code. Only handles the rank-2 case,
    which is all this project's real weight tensors ever are."""
    row_count = len(weights)
    column_count = len(weights[0])

    if axis == 1:
        channel_count = column_count
        channel_values = [
            [weights[row][column] for row in range(row_count)]
            for column in range(channel_count)
        ]
    else:
        channel_count = row_count
        channel_values = [list(weights[row]) for row in range(channel_count)]

    scale = []
    quantized_channels = []
    for values in channel_values:
        max_abs = max(abs(value) for value in values)
        channel_scale = max_abs / 127.0 if max_abs > 0.0 else 0.0
        divisor = channel_scale if channel_scale > 0.0 else 1.0
        quantized_channels.append(
            [
                max(-127, min(127, round(value / divisor)))
                for value in values
            ]
        )
        scale.append(channel_scale)

    quantized = [[0] * column_count for _ in range(row_count)]
    for channel_index, values in enumerate(quantized_channels):
        for position, value in enumerate(values):
            if axis == 1:
                quantized[position][channel_index] = value
            else:
                quantized[channel_index][position] = value

    return quantized, scale


class QuantizePerChannelTest(unittest.TestCase):
    def test_matches_an_independent_reference_implementation(self) -> None:
        weights = np.array(
            [
                [57.0, 58.0, -59.0, 60.0],
                [69.0, -70.0, 71.0, 72.0],
                [81.0, 82.0, 83.0, -84.0],
                [-93.0, 94.0, 95.0, 96.0],
            ],
            dtype=np.float32,
        )

        for axis in (0, 1):
            with self.subTest(axis=axis):
                quantized, scale = quantize_per_channel(weights, axis)
                expected_quantized, expected_scale = (
                    _reference_quantize_per_channel(weights.tolist(), axis)
                )

                self.assertEqual(quantized.dtype, np.dtype(np.int8))
                self.assertEqual(quantized.tolist(), expected_quantized)
                np.testing.assert_allclose(
                    scale, expected_scale, rtol=0, atol=1e-6
                )

    def test_channel_maximum_quantizes_to_the_boundary(self) -> None:
        weights = np.array([[10.0, -20.0], [30.0, -40.0]], dtype=np.float32)

        quantized, scale = quantize_per_channel(weights, axis=1)

        # Column 0's largest magnitude is 30, column 1's is 40; each
        # column's own maximum must land on exactly +-127, never one
        # short (126) or one over into the unrepresentable -128.
        self.assertEqual(quantized[1, 0], 127)
        self.assertEqual(quantized[1, 1], -127)
        np.testing.assert_allclose(scale, [30.0 / 127.0, 40.0 / 127.0])

    def test_all_zero_channel_quantizes_to_zero_without_dividing_by_zero(
        self,
    ) -> None:
        weights = np.array(
            [[0.0, 5.0], [0.0, -5.0]],
            dtype=np.float32,
        )

        quantized, scale = quantize_per_channel(weights, axis=1)

        self.assertEqual(quantized[:, 0].tolist(), [0, 0])
        self.assertEqual(scale[0], 0.0)

        # Column 1's own values set its scale, so they occupy the full
        # int8 range regardless of their absolute magnitude -- the same
        # boundary behaviour as test_channel_maximum_quantizes_to_the_
        # boundary above, not a smaller, "gentler" quantization just
        # because 5.0 looks like a small number in isolation.
        self.assertEqual(quantized[:, 1].tolist(), [127, -127])

    def test_dequantized_values_stay_within_half_a_scale_step(self) -> None:
        rng = np.random.default_rng(20260905)
        weights = rng.normal(scale=3.0, size=(11, 7)).astype(np.float32)

        for axis in (0, 1):
            with self.subTest(axis=axis):
                quantized, scale = quantize_per_channel(weights, axis)
                broadcast_shape = [1, 1]
                broadcast_shape[axis] = weights.shape[axis]
                dequantized = quantized.astype(np.float64) * scale.reshape(
                    broadcast_shape
                )

                tolerance = (scale / 2.0).reshape(broadcast_shape) + 1e-9
                error = np.abs(dequantized - weights)
                self.assertTrue(
                    bool(np.all(error <= np.broadcast_to(tolerance, error.shape))),
                    "a dequantized value exceeded half its channel's scale",
                )

    def test_rejects_an_axis_outside_the_tensor_rank(self) -> None:
        weights = np.zeros((2, 3), dtype=np.float32)

        with self.assertRaises(ValueError):
            quantize_per_channel(weights, axis=2)


class QuantizedExportTest(unittest.TestCase):
    def test_transformer_weights_quantize_with_the_output_channel_axis(
        self,
    ) -> None:
        model = make_model()
        _, records = collect_tensor_records(
            model,
            encode_tensor=fake_encode,
            tensors_equal=fake_equal,
            quantize_transformer_weights=True,
        )
        records_by_name = {record.name: record for record in records}

        c_attn = records_by_name["transformer.h.0.attn.c_attn.weight"]
        self.assertEqual(c_attn.dtype, INT8)
        self.assertEqual(len(c_attn.payload), 4 * 12)

        c_attn_scale = records_by_name[
            "transformer.h.0.attn.c_attn.weight.quant_scale"
        ]
        self.assertEqual(c_attn_scale.shape, (12,))
        self.assertEqual(len(c_attn_scale.payload), 12 * 4)

        for role, out_features in (
            ("attn.c_attn.weight", 12),
            ("attn.c_proj.weight", 4),
            ("mlp.c_fc.weight", 16),
            ("mlp.c_proj.weight", 4),
        ):
            name = f"transformer.h.0.{role}"
            with self.subTest(role=role):
                self.assertEqual(records_by_name[name].dtype, INT8)
                self.assertEqual(
                    records_by_name[f"{name}.quant_scale"].shape,
                    (out_features,),
                )

        # Biases, LayerNorm and position embeddings are never
        # quantized, and neither is the tied embedding unless
        # separately requested.
        for name in (
            "transformer.wte.weight",
            "transformer.wpe.weight",
            "transformer.h.0.ln_1.weight",
            "transformer.h.0.ln_1.bias",
            "transformer.h.0.attn.c_attn.bias",
            "transformer.h.0.attn.c_proj.bias",
            "transformer.h.0.mlp.c_fc.bias",
            "transformer.h.0.mlp.c_proj.bias",
            "transformer.ln_f.weight",
            "transformer.ln_f.bias",
        ):
            with self.subTest(name=name):
                self.assertNotIn(f"{name}.quant_scale", records_by_name)
                self.assertNotEqual(records_by_name[name].dtype, INT8)

    def test_tied_embedding_quantizes_with_the_vocabulary_axis(self) -> None:
        model = make_model()
        _, records = collect_tensor_records(
            model,
            encode_tensor=fake_encode,
            tensors_equal=fake_equal,
            quantize_transformer_weights=True,
            quantize_tied_embedding=True,
        )
        records_by_name = {record.name: record for record in records}

        wte = records_by_name["transformer.wte.weight"]
        self.assertEqual(wte.dtype, INT8)
        self.assertEqual(wte.shape, (8, 4))

        # The tied embedding's channel axis is the vocabulary (its
        # rows), not its embedding width (its columns) -- the opposite
        # of every other quantized weight in this model, because it is
        # read row-wise for both embedding lookup and the tied LM-head
        # projection. A scale of length 4 here (the embedding width)
        # instead of 8 would mean the wrong axis was quantized.
        wte_scale = records_by_name["transformer.wte.weight.quant_scale"]
        self.assertEqual(wte_scale.shape, (8,))

    def test_tied_embedding_stays_fp32_without_the_flag(self) -> None:
        model = make_model()
        _, records = collect_tensor_records(
            model,
            encode_tensor=fake_encode,
            tensors_equal=fake_equal,
            quantize_transformer_weights=True,
            quantize_tied_embedding=False,
        )
        records_by_name = {record.name: record for record in records}

        self.assertNotEqual(records_by_name["transformer.wte.weight"].dtype, INT8)
        self.assertNotIn(
            "transformer.wte.weight.quant_scale", records_by_name
        )

    def test_quantized_export_round_trips_through_the_binary_writer(
        self,
    ) -> None:
        model = make_model()

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "quantized-gpt2.bin"
            summary = export_model(
                model,
                output_path,
                encode_tensor=fake_encode,
                tensors_equal=fake_equal,
                quantize_transformer_weights=True,
                quantize_tied_embedding=True,
            )
            fp32_summary = export_model(
                model,
                Path(directory) / "fp32-gpt2.bin",
                encode_tensor=fake_encode,
                tensors_equal=fake_equal,
            )

        # Four transformer weights per layer plus the tied embedding
        # are quantized; each gains one extra record for its scale.
        self.assertEqual(summary.tensor_count, fp32_summary.tensor_count + 5)
        # The two exports represent the same trained model, so this
        # count must not drift just because some weights are stored
        # differently.
        self.assertEqual(summary.parameter_count, fp32_summary.parameter_count)
        self.assertLess(summary.file_size, fp32_summary.file_size)


if __name__ == "__main__":
    unittest.main()
