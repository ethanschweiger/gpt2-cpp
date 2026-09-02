#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace gpt2 {

class Tensor {
public:
    using Shape = std::vector<std::size_t>;

    explicit Tensor(Shape shape, float fill_value = 0.0F);
    Tensor(Shape shape, std::vector<float> values);

    std::size_t rank() const;
    std::size_t numel() const;
    const Shape& shape() const;

    float* data();
    const float* data() const;

    float& at(std::size_t flat_index);
    const float& at(std::size_t flat_index) const;

    float& at(std::span<const std::size_t> indices);
    const float& at(std::span<const std::size_t> indices) const;

private:
    std::size_t offset(
        std::span<const std::size_t> indices) const;

    Shape shape_m;
    std::vector<float> data_m;
};

}  // namespace gpt2
