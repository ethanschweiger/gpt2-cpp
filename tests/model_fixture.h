#pragma once

// Builds a small GPT-2 checkpoint in memory so that model-level tests
// can run without the real model files. Every value is derived from its
// position, so the same fixture can be rebuilt in PyTorch to produce
// reference expectations.

#include "gpt2/checkpoint.h"

#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace gpt2_test {

constexpr std::uint32_t fixture_vocabulary_size = 7;
constexpr std::uint32_t fixture_context_length = 5;
constexpr std::uint32_t fixture_embedding_size = 4;
constexpr std::uint32_t fixture_head_count = 2;
constexpr std::uint32_t fixture_layer_count = 2;

// Two parameterizations of the same architecture. The alternate one is
// chosen because its greedy continuation changes between steps, which
// the standard one's does not.
constexpr std::size_t standard_index_multiplier = 7;
constexpr std::size_t alternate_index_multiplier = 3;

enum class ValueKind {
    token_embedding,
    weight,
    layer_norm_weight,
    bias,
};

struct TensorFixture {
    std::string name;
    std::vector<std::uint64_t> shape;
    std::vector<float> values;
};

using Bytes = std::vector<unsigned char>;

inline std::size_t element_count(
    const std::vector<std::uint64_t>& shape
) {
    std::size_t result = 1;
    for (const std::uint64_t dimension : shape) {
        result *= static_cast<std::size_t>(dimension);
    }
    return result;
}

inline float fixture_value(
    std::size_t index,
    std::size_t seed,
    ValueKind kind,
    std::size_t index_multiplier
) {
    const std::size_t encoded =
        (index * index_multiplier + seed * std::size_t{5}) %
        std::size_t{17};
    const int raw = static_cast<int>(encoded) - 8;
    const float value = static_cast<float>(raw);

    switch (kind) {
    case ValueKind::token_embedding:
        return value / 32.0F;
    case ValueKind::weight:
        return value / 64.0F;
    case ValueKind::layer_norm_weight:
        return 1.0F + value / 64.0F;
    case ValueKind::bias:
        return value / 128.0F;
    }

    throw std::logic_error("unknown fixture value kind");
}

inline TensorFixture make_tensor(
    std::string name,
    std::vector<std::uint64_t> shape,
    ValueKind kind,
    std::size_t seed,
    std::size_t index_multiplier = standard_index_multiplier
) {
    std::vector<float> values(element_count(shape));
    for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] = fixture_value(
            index,
            seed,
            kind,
            index_multiplier
        );
    }

    return {
        std::move(name),
        std::move(shape),
        std::move(values),
    };
}

inline void append_model_tensor(
    std::vector<TensorFixture>& tensors,
    std::string name,
    std::vector<std::uint64_t> shape,
    ValueKind kind,
    std::size_t index_multiplier
) {
    const std::size_t seed = tensors.size() + 1;
    tensors.push_back(make_tensor(
        std::move(name),
        std::move(shape),
        kind,
        seed,
        index_multiplier
    ));
}

inline std::vector<TensorFixture> make_model_tensors(
    std::size_t index_multiplier = standard_index_multiplier
) {
    std::vector<TensorFixture> tensors;
    tensors.reserve(28);

    append_model_tensor(
        tensors,
        "transformer.wte.weight",
        {7, 4},
        ValueKind::token_embedding,
        index_multiplier
    );
    append_model_tensor(
        tensors,
        "transformer.wpe.weight",
        {5, 4},
        ValueKind::weight,
        index_multiplier
    );

    for (std::size_t layer = 0; layer < 2; ++layer) {
        const std::string prefix =
            "transformer.h." + std::to_string(layer);

        append_model_tensor(
            tensors,
            prefix + ".ln_1.weight",
            {4},
            ValueKind::layer_norm_weight,
            index_multiplier
        );
        append_model_tensor(
            tensors,
            prefix + ".ln_1.bias",
            {4},
            ValueKind::bias,
            index_multiplier
        );
        append_model_tensor(
            tensors,
            prefix + ".attn.c_attn.weight",
            {4, 12},
            ValueKind::weight,
            index_multiplier
        );
        append_model_tensor(
            tensors,
            prefix + ".attn.c_attn.bias",
            {12},
            ValueKind::bias,
            index_multiplier
        );
        append_model_tensor(
            tensors,
            prefix + ".attn.c_proj.weight",
            {4, 4},
            ValueKind::weight,
            index_multiplier
        );
        append_model_tensor(
            tensors,
            prefix + ".attn.c_proj.bias",
            {4},
            ValueKind::bias,
            index_multiplier
        );
        append_model_tensor(
            tensors,
            prefix + ".ln_2.weight",
            {4},
            ValueKind::layer_norm_weight,
            index_multiplier
        );
        append_model_tensor(
            tensors,
            prefix + ".ln_2.bias",
            {4},
            ValueKind::bias,
            index_multiplier
        );
        append_model_tensor(
            tensors,
            prefix + ".mlp.c_fc.weight",
            {4, 16},
            ValueKind::weight,
            index_multiplier
        );
        append_model_tensor(
            tensors,
            prefix + ".mlp.c_fc.bias",
            {16},
            ValueKind::bias,
            index_multiplier
        );
        append_model_tensor(
            tensors,
            prefix + ".mlp.c_proj.weight",
            {16, 4},
            ValueKind::weight,
            index_multiplier
        );
        append_model_tensor(
            tensors,
            prefix + ".mlp.c_proj.bias",
            {4},
            ValueKind::bias,
            index_multiplier
        );
    }

    append_model_tensor(
        tensors,
        "transformer.ln_f.weight",
        {4},
        ValueKind::layer_norm_weight,
        index_multiplier
    );
    append_model_tensor(
        tensors,
        "transformer.ln_f.bias",
        {4},
        ValueKind::bias,
        index_multiplier
    );

    return tensors;
}

inline TensorFixture& find_tensor(
    std::vector<TensorFixture>& tensors,
    std::string_view name
) {
    for (TensorFixture& tensor : tensors) {
        if (tensor.name == name) {
            return tensor;
        }
    }

    throw std::logic_error("fixture tensor not found");
}

inline void append_u32(Bytes& bytes, std::uint32_t value) {
    for (unsigned int shift = 0; shift < 32; shift += 8) {
        bytes.push_back(static_cast<unsigned char>(value >> shift));
    }
}

inline void append_u64(Bytes& bytes, std::uint64_t value) {
    for (unsigned int shift = 0; shift < 64; shift += 8) {
        bytes.push_back(static_cast<unsigned char>(value >> shift));
    }
}

inline void append_float(Bytes& bytes, float value) {
    append_u32(bytes, std::bit_cast<std::uint32_t>(value));
}

inline Bytes make_checkpoint(
    const std::vector<TensorFixture>& tensors
) {
    Bytes bytes{
        'G', 'P', 'T', '2', 'C', 'P', 'P', '\0'
    };
    append_u32(bytes, 1);
    append_u32(bytes, 64);
    append_u32(bytes, 0x01020304U);
    append_u32(bytes, 0);
    append_u32(bytes, static_cast<std::uint32_t>(tensors.size()));
    append_u32(bytes, fixture_vocabulary_size);
    append_u32(bytes, fixture_context_length);
    append_u32(bytes, fixture_embedding_size);
    append_u32(bytes, fixture_head_count);
    append_u32(bytes, fixture_layer_count);
    for (int index = 0; index < 4; ++index) {
        append_u32(bytes, 0);
    }

    for (const TensorFixture& tensor : tensors) {
        append_u32(
            bytes,
            static_cast<std::uint32_t>(tensor.name.size())
        );
        append_u32(bytes, 1);
        append_u32(
            bytes,
            static_cast<std::uint32_t>(tensor.shape.size())
        );
        append_u32(bytes, 0);
        append_u64(
            bytes,
            static_cast<std::uint64_t>(tensor.values.size())
        );
        append_u64(
            bytes,
            static_cast<std::uint64_t>(tensor.values.size()) * 4U
        );

        for (const std::uint64_t dimension : tensor.shape) {
            append_u64(bytes, dimension);
        }
        bytes.insert(
            bytes.end(),
            tensor.name.begin(),
            tensor.name.end()
        );
        for (const float value : tensor.values) {
            append_float(bytes, value);
        }
    }

    return bytes;
}

inline std::filesystem::path unique_temporary_path(
    std::string_view prefix
) {
    static std::uint64_t counter = 0;
    const auto timestamp = std::chrono::steady_clock::now()
        .time_since_epoch()
        .count();
    return std::filesystem::temp_directory_path() /
        (std::string(prefix) + std::to_string(timestamp) +
         "-" + std::to_string(++counter) + ".bin");
}

class TemporaryCheckpoint {
public:
    explicit TemporaryCheckpoint(
        const Bytes& bytes,
        std::string_view prefix = "gpt2-model-test-"
    )
        : path_m(unique_temporary_path(prefix)) {
        std::ofstream output(path_m, std::ios::binary);
        output.write(
            reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size())
        );
        if (!output) {
            throw std::runtime_error(
                "could not write temporary model checkpoint"
            );
        }
    }

    TemporaryCheckpoint(const TemporaryCheckpoint&) = delete;
    TemporaryCheckpoint& operator=(const TemporaryCheckpoint&) = delete;

    ~TemporaryCheckpoint() {
        remove();
    }

    const std::filesystem::path& path() const {
        return path_m;
    }

    void remove() {
        std::error_code error;
        std::filesystem::remove(path_m, error);
    }

private:
    std::filesystem::path path_m;
};

}  // namespace gpt2_test
