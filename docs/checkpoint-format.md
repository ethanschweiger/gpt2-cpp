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

Version 1 writers emit only FP32 tensors. Future versions may define int8
quantization metadata and additional data-type codes.

## Required invariants

A valid version 1 checkpoint satisfies all of these conditions:

- The tensor count exactly matches the number of records.
- Every tensor name is non-empty and unique.
- Every rank and dimension is greater than zero.
- The element count equals the product of all dimensions.
- For FP32, the payload size equals `element count * 4`.
- The embedding size is divisible by the attention-head count.
- All version 1 flags and reserved fields are zero.
- No record or payload extends beyond the end of the file.
- No trailing bytes follow the final payload.

Readers must reject unsupported versions, data types, invalid metadata,
arithmetic overflow, truncated data, duplicate tensor names, and trailing data.

## Portability rule

Writers and readers must encode and decode each field explicitly. They must not
write or read an in-memory C++ structure as a raw byte block because structure
padding and native endianness are platform-dependent.
