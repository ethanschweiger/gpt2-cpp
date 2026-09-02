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
    float32 = 1
};

}  // namespace gpt2::checkpoint_format
