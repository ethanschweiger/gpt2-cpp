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

private:
    using TensorMap = std::unordered_map<std::string, Tensor>;

    Checkpoint(ModelConfig config, TensorMap tensors);

    ModelConfig config_m;
    TensorMap tensors_m;

    friend Checkpoint load_checkpoint(
        const std::filesystem::path& path
    );
};

Checkpoint load_checkpoint(const std::filesystem::path& path);

}  // namespace gpt2
