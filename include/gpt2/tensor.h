#pragma once

#include <cstddef>
#include <cstdint>
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

    void reshape(Shape new_shape);

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

// Raw int8 storage for a symmetric, per-channel-quantized weight
// tensor. Values are always required to fall in [-127, 127]; -128 has
// no symmetric-quantization counterpart and is rejected wherever an
// Int8Tensor is constructed from external data.
//
// This type only holds bytes and their shape — it offers no
// arithmetic of its own. gpt2::quantized_linear (tensor_ops.h)
// dequantizes and multiplies; see docs/quantization.md for the scheme.
class Int8Tensor {
public:
    using Shape = std::vector<std::size_t>;

    Int8Tensor(Shape shape, std::vector<std::int8_t> values);

    std::size_t rank() const;
    std::size_t numel() const;
    const Shape& shape() const;

    std::int8_t* data();
    const std::int8_t* data() const;

private:
    Shape shape_m;
    std::vector<std::int8_t> data_m;
};

}  // namespace gpt2
