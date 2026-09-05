# GPT-2 CPP Checkpoint Format

This document specifies version 1 of the binary checkpoint format shared by
the Python exporter and the C++ inference runtime.

## Encoding rules

- All integers use unsigned little-endian encoding.
- `uint32` and `uint64` contain exactly 32 and 64 bits, respectively.
- Tensor payloads use little-endian IEEE-754 binary32 values.
- Tensor elements use contiguous row-major order.
- Tensor names use UTF-8 bytes without a null terminator.
- Fields are written individually. The file contains no compiler-generated
  structure padding.

## Global header

Every checkpoint begins with this fixed 64-byte header:

| Offset | Size | Type | Field | Version 1 value |
| ---: | ---: | --- | --- | --- |
| 0 | 8 | bytes | magic | `GPT2CPP\0` |
| 8 | 4 | uint32 | version | `1` |
| 12 | 4 | uint32 | header size | `64` |
| 16 | 4 | uint32 | endian marker | `0x01020304` |
| 20 | 4 | uint32 | flags | `0` |
| 24 | 4 | uint32 | tensor count | Number of tensor records |
| 28 | 4 | uint32 | vocabulary size | Model configuration |
| 32 | 4 | uint32 | context length | Model configuration |
| 36 | 4 | uint32 | embedding size | Model configuration |
| 40 | 4 | uint32 | attention-head count | Model configuration |
| 44 | 4 | uint32 | transformer-layer count | Model configuration |
| 48 | 16 | uint32[4] | reserved | All zero |

The magic bytes identify this file format. The version and header-size fields
allow readers to reject layouts they do not support. The endian marker confirms
that the writer used the required byte order.

## Tensor records

Tensor records immediately follow the global header. Each record starts with a
fixed 32-byte header:

| Relative offset | Size | Type | Field |
| ---: | ---: | --- | --- |
| 0 | 4 | uint32 | Tensor-name length in bytes |
| 4 | 4 | uint32 | Data-type code |
| 8 | 4 | uint32 | Rank |
| 12 | 4 | uint32 | Flags |
| 16 | 8 | uint64 | Element count |
| 24 | 8 | uint64 | Payload size in bytes |

The fixed header is followed, without padding, by:

1. `rank` dimensions, each encoded as a `uint64`.
2. Exactly `name length` UTF-8 bytes.
3. Exactly `payload size` tensor-data bytes.

The next tensor record begins immediately after the previous payload. The file
ends immediately after the final payload.

## Data types

| Code | Meaning | Bytes per element |
| ---: | --- | ---: |
| 1 | FP32 | 4 |
| 2 | INT8 | 1 |

Adding INT8 did not require a new format version: the container (the global
header, and every field of a tensor record) is unchanged, and a version 1
reader already rejects any `data_type` value other than `1` with a clear
error, so an INT8-bearing file is safely refused rather than misread by an
older reader. Only the set of data-type codes a *version 1* reader is
willing to accept has grown.

## Quantization

An INT8 tensor named `<name>` represents a symmetric, per-channel-quantized
weight: each channel `i` was quantized as `Q_i = round(W_i / s_i)`, `Q_i in
[-127, 127]`, with the corresponding scale `s_i = max(|W_i|) / 127`, and is
approximately recovered as `W_i ≈ s_i * Q_i`. The value `-128` has no
symmetric-quantization counterpart under this scheme and is rejected wherever
it appears in an INT8 payload.

A checkpoint that contains `<name>` as an INT8 tensor must also contain a
tensor named `<name>.quant_scale` — an ordinary FP32, rank-1 tensor whose
length equals one of `<name>`'s own dimensions. That length is the *number of
channels*; which of `<name>`'s dimensions is "the channel axis" is
deliberately not recorded in the checkpoint itself, because a matrix with two
equal dimensions would make that ambiguous from shape alone even if it were.
It is instead a fixed property of what `<name>` represents, decided by
whichever model-loading code loads that specific tensor by name — the same
place that already knows, independent of quantization, what shape `<name>`
is supposed to have. The scale record may appear before or after its INT8
tensor in the file; the loader accepts either order.

See [Quantization](quantization.md) for GPT-2's own per-tensor axis choices
and how `tools/export_gpt2.py` produces an INT8 checkpoint.

A checkpoint may freely mix FP32 and INT8 tensors. A given tensor name is
one or the other, never both, and `<name>.quant_scale` is always FP32
regardless of what `<name>` is.

## Required invariants

A valid checkpoint satisfies all of these conditions:

- The tensor count exactly matches the number of records.
- All five model-configuration values are greater than zero.
- Every tensor name is non-empty and unique, across both data types.
- Every rank and dimension is greater than zero.
- The element count equals the product of all dimensions.
- For FP32, the payload size equals `element count * 4`.
- For INT8, the payload size equals `element count * 1`, and every payload
  byte is a value other than `-128`.
- Every INT8 tensor `<name>` is paired with an FP32, rank-1 tensor named
  `<name>.quant_scale` whose length matches one of `<name>`'s dimensions.
- The embedding size is divisible by the attention-head count.
- All version 1 flags and reserved fields are zero.
- No record or payload extends beyond the end of the file.
- No trailing bytes follow the final payload.

Readers must reject unsupported versions, data types, invalid metadata,
arithmetic overflow, truncated data, duplicate tensor names, and trailing data.

## C++ loader resource limits

The reference C++ loader applies defensive limits before allocating memory:

- At most 4,096 tensor records.
- At most 8 dimensions per tensor.
- At most 1,024 UTF-8 bytes per tensor name.
- At most 512 MiB of payload data per tensor.
- At most 8 GiB of payload data across the checkpoint.

These limits comfortably cover the GPT-2 model family while preventing a
malformed or sparse file from requesting allocations based only on extreme
metadata values.

## Portability rule

Writers and readers must encode and decode each field explicitly. They must not
write or read an in-memory C++ structure as a raw byte block because structure
padding and native endianness are platform-dependent.
