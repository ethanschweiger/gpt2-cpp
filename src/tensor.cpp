#include "gpt2/tensor.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace gpt2 {

namespace {

std::size_t checked_numel(const Tensor::Shape& shape) {
    if (shape.empty()) {
        throw std::invalid_argument(
            "tensor shape must have at least one dimension"
        );
    }

    std::size_t element_count = 1;

    for (const std::size_t dimension : shape) {
        if (dimension == 0) {
            throw std::invalid_argument(
                "tensor dimensions must be greater than zero"
            );
        }

        if (element_count >
            std::numeric_limits<std::size_t>::max() / dimension) {
            throw std::overflow_error(
                "tensor element count is too large"
            );
        }

        element_count *= dimension;
    }

    return element_count;
}

}  // namespace

Tensor::Tensor(Shape shape, float fill_value)
    : shape_m(std::move(shape)),
      data_m(checked_numel(shape_m), fill_value) {}

Tensor::Tensor(Shape shape, std::vector<float> values)
    : shape_m(std::move(shape)),
      data_m(std::move(values)) {
    if (data_m.size() != checked_numel(shape_m)) {
        throw std::invalid_argument(
            "tensor data size does not match its shape"
        );
    }
}

std::size_t Tensor::rank() const {
    return shape_m.size();
}

std::size_t Tensor::numel() const {
    return data_m.size();
}

const Tensor::Shape& Tensor::shape() const {
    return shape_m;
}

void Tensor::reshape(Shape new_shape) {
    const std::size_t new_element_count = checked_numel(new_shape);

    if (new_element_count != numel()) {
        throw std::invalid_argument(
            "cannot reshape tensor to a different element count"
        );
    }

    shape_m = std::move(new_shape);
}

float* Tensor::data() {
    return data_m.data();
}

const float* Tensor::data() const {
    return data_m.data();
}

float& Tensor::at(std::size_t flat_index) {
    return data_m.at(flat_index);
}

const float& Tensor::at(std::size_t flat_index) const {
    return data_m.at(flat_index);
}

float& Tensor::at(std::span<const std::size_t> indices) {
    return at(offset(indices));
}

const float& Tensor::at(
    std::span<const std::size_t> indices
) const {
    return at(offset(indices));
}

std::size_t Tensor::offset(
    std::span<const std::size_t> indices
) const {
    if (indices.size() != rank()) {
        throw std::invalid_argument(
            "tensor index count does not match its rank"
        );
    }

    std::size_t flat_index = 0;

    for (std::size_t axis = 0; axis < rank(); ++axis) {
        if (indices[axis] >= shape_m[axis]) {
            throw std::out_of_range(
                "tensor index is outside its shape"
            );
        }

        flat_index =
            flat_index * shape_m[axis] + indices[axis];
    }

    return flat_index;
}

}  // namespace gpt2
