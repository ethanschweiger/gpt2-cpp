#pragma once

#include <array>
#include <cstdint>

namespace gpt2::checkpoint_format {

inline constexpr std::array<char, 8> magic{
    'G', 'P', 'T', '2', 'C', 'P', 'P', '\0'
};

inline constexpr std::uint32_t version = 1;
inline constexpr std::uint32_t header_size = 64;
inline constexpr std::uint32_t endian_marker = 0x01020304U;

enum class DataType : std::uint32_t {
    float32 = 1,

    // A symmetric, per-channel-quantized int8 tensor. Every int8
    // tensor named "<name>" must be accompanied by a float32,
    // rank-1 tensor named "<name>.quant_scale" whose length equals
    // one of the int8 tensor's own dimensions — the loader enforces
    // this pairing generically, and model-loading code (which already
    // knows what each named tensor represents) decides which
    // dimension that is. See docs/checkpoint-format.md and
    // docs/quantization.md.
    int8 = 2
};

}  // namespace gpt2::checkpoint_format
