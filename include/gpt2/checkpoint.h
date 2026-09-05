#pragma once

#include "gpt2/tensor.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>

namespace gpt2 {

struct ModelConfig {
    std::uint32_t vocab_size;
    std::uint32_t context_length;
    std::uint32_t embedding_size;
    std::uint32_t head_count;
    std::uint32_t layer_count;
};

class Checkpoint {
public:
    const ModelConfig& config() const;
    std::size_t tensor_count() const;

    bool contains(std::string_view name) const;
    const Tensor& tensor(std::string_view name) const;

    // A tensor is stored as either float32 (above) or int8 (below),
    // never both, so a given name answers exactly one of these two
    // pairs. An int8 tensor's per-channel scale is itself an ordinary
    // float32 tensor, found by appending ".quant_scale" to its name
    // and looking it up with contains()/tensor() above.
    bool contains_int8(std::string_view name) const;
    const Int8Tensor& int8_tensor(std::string_view name) const;

private:
    using TensorMap = std::unordered_map<std::string, Tensor>;
    using Int8TensorMap = std::unordered_map<std::string, Int8Tensor>;

    Checkpoint(
        ModelConfig config,
        TensorMap tensors,
        Int8TensorMap int8_tensors
    );

    ModelConfig config_m;
    TensorMap tensors_m;
    Int8TensorMap int8_tensors_m;

    friend Checkpoint load_checkpoint(
        const std::filesystem::path& path
    );
};

Checkpoint load_checkpoint(const std::filesystem::path& path);

}  // namespace gpt2
