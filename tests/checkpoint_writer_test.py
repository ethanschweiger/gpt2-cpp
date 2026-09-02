import struct
import tempfile
from pathlib import Path
import unittest

from checkpoint_writer import (
    ModelConfig,
    TensorRecord,
    write_checkpoint,
)


EXPECTED_MAGIC = b"GPT2CPP\0"
EXPECTED_VERSION = 1
EXPECTED_HEADER_SIZE = 64
EXPECTED_ENDIAN_MARKER = 0x01020304
EXPECTED_FLOAT32 = 1

EXPECTED_GLOBAL_HEADER = struct.Struct("<8s14I")
EXPECTED_TENSOR_HEADER = struct.Struct("<IIIIQQ")


def fp32_payload(*values: float) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


class CheckpointWriterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ModelConfig(
            vocab_size=4,
            context_length=8,
            embedding_size=4,
            head_count=2,
            layer_count=1,
        )

    def test_writes_expected_binary_layout(self) -> None:
        tensor = TensorRecord(
            name="weight",
            shape=(2, 2),
            payload=fp32_payload(1.0, 2.0, 3.0, 4.0),
        )

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "tiny.bin"
            write_checkpoint(output_path, self.config, [tensor])
            contents = output_path.read_bytes()

        self.assertEqual(EXPECTED_GLOBAL_HEADER.size, 64)
        self.assertEqual(EXPECTED_TENSOR_HEADER.size, 32)

        global_header = EXPECTED_GLOBAL_HEADER.unpack_from(contents, 0)
        self.assertEqual(
            global_header,
            (
                EXPECTED_MAGIC,
                EXPECTED_VERSION,
                EXPECTED_HEADER_SIZE,
                EXPECTED_ENDIAN_MARKER,
                0,
                1,
                4,
                8,
                4,
                2,
                1,
                0,
                0,
                0,
                0,
            ),
        )

        offset = EXPECTED_GLOBAL_HEADER.size
        tensor_header = EXPECTED_TENSOR_HEADER.unpack_from(contents, offset)
        self.assertEqual(tensor_header, (6, EXPECTED_FLOAT32, 2, 0, 4, 16))
        offset += EXPECTED_TENSOR_HEADER.size

        dimensions = struct.unpack_from("<2Q", contents, offset)
        self.assertEqual(dimensions, (2, 2))
        offset += struct.calcsize("<2Q")

        name = contents[offset : offset + 6]
        self.assertEqual(name, b"weight")
        offset += len(name)

        payload = contents[offset : offset + 16]
        self.assertEqual(struct.unpack("<4f", payload), (1.0, 2.0, 3.0, 4.0))
        offset += len(payload)

        self.assertEqual(offset, len(contents))

    def test_writes_multiple_tensor_records_in_order(self) -> None:
        tensors = [
            TensorRecord("a", (1,), fp32_payload(1.0)),
            TensorRecord("beta-β", (1, 2), fp32_payload(2.0, 3.0)),
        ]

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "multiple.bin"
            write_checkpoint(output_path, self.config, tensors)
            contents = output_path.read_bytes()

        global_header = EXPECTED_GLOBAL_HEADER.unpack_from(contents, 0)
        self.assertEqual(global_header[5], 2)

        offset = EXPECTED_GLOBAL_HEADER.size

        first_header = EXPECTED_TENSOR_HEADER.unpack_from(contents, offset)
        self.assertEqual(first_header, (1, EXPECTED_FLOAT32, 1, 0, 1, 4))
        offset += EXPECTED_TENSOR_HEADER.size
        self.assertEqual(struct.unpack_from("<Q", contents, offset), (1,))
        offset += struct.calcsize("<Q")
        self.assertEqual(contents[offset : offset + 1], b"a")
        offset += 1
        self.assertEqual(struct.unpack_from("<f", contents, offset), (1.0,))
        offset += struct.calcsize("<f")

        second_name = "beta-β".encode("utf-8")
        second_header = EXPECTED_TENSOR_HEADER.unpack_from(contents, offset)
        self.assertEqual(
            second_header,
            (len(second_name), EXPECTED_FLOAT32, 2, 0, 2, 8),
        )
        offset += EXPECTED_TENSOR_HEADER.size
        self.assertEqual(struct.unpack_from("<2Q", contents, offset), (1, 2))
        offset += struct.calcsize("<2Q")
        self.assertEqual(contents[offset : offset + len(second_name)], second_name)
        offset += len(second_name)
        self.assertEqual(struct.unpack_from("<2f", contents, offset), (2.0, 3.0))
        offset += struct.calcsize("<2f")

        self.assertEqual(offset, len(contents))

    def test_rejects_invalid_model_config(self) -> None:
        invalid_config = ModelConfig(
            vocab_size=4,
            context_length=8,
            embedding_size=3,
            head_count=2,
            layer_count=1,
        )

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "invalid.bin"
            with self.assertRaisesRegex(ValueError, "divisible"):
                write_checkpoint(output_path, invalid_config, [])
            self.assertFalse(output_path.exists())

    def test_rejects_duplicate_tensor_names(self) -> None:
        tensors = [
            TensorRecord("weight", (1,), fp32_payload(1.0)),
            TensorRecord("weight", (1,), fp32_payload(2.0)),
        ]

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "duplicate.bin"
            with self.assertRaisesRegex(ValueError, "duplicate"):
                write_checkpoint(output_path, self.config, tensors)
            self.assertFalse(output_path.exists())

    def test_rejects_invalid_tensor_shapes(self) -> None:
        cases = (
            TensorRecord("empty", (), b""),
            TensorRecord("zero", (2, 0), b""),
        )

        with tempfile.TemporaryDirectory() as directory:
            for tensor in cases:
                with self.subTest(name=tensor.name):
                    output_path = Path(directory) / f"{tensor.name}.bin"
                    with self.assertRaises(ValueError):
                        write_checkpoint(output_path, self.config, [tensor])
                    self.assertFalse(output_path.exists())

    def test_rejects_shape_overflow(self) -> None:
        tensor = TensorRecord("huge", ((1 << 64) - 1, 2), b"")

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "overflow.bin"
            with self.assertRaises(OverflowError):
                write_checkpoint(output_path, self.config, [tensor])
            self.assertFalse(output_path.exists())

    def test_rejects_incorrect_payload_size(self) -> None:
        tensor = TensorRecord("weight", (2, 2), fp32_payload(1.0))

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "payload.bin"
            with self.assertRaisesRegex(ValueError, "requires 16"):
                write_checkpoint(output_path, self.config, [tensor])
            self.assertFalse(output_path.exists())

    def test_rejects_empty_tensor_name(self) -> None:
        tensor = TensorRecord("", (1,), fp32_payload(1.0))

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "name.bin"
            with self.assertRaisesRegex(ValueError, "must not be empty"):
                write_checkpoint(output_path, self.config, [tensor])
            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
