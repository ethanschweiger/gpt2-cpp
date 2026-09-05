#include "gpt2/tensor.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace gpt2 {

namespace {

std::size_t checked_numel(const Int8Tensor::Shape& shape) {
    if (shape.empty()) {
        throw std::invalid_argument(
            "int8 tensor shape must have at least one dimension"
        );
    }

    std::size_t element_count = 1;

    for (const std::size_t dimension : shape) {
        if (dimension == 0) {
            throw std::invalid_argument(
                "int8 tensor dimensions must be greater than zero"
            );
        }

        if (element_count >
            std::numeric_limits<std::size_t>::max() / dimension) {
            throw std::overflow_error(
                "int8 tensor element count is too large"
            );
        }

        element_count *= dimension;
    }

    return element_count;
}

void require_symmetric_range(const std::vector<std::int8_t>& values) {
    for (const std::int8_t value : values) {
        if (value == std::numeric_limits<std::int8_t>::min()) {
            throw std::invalid_argument(
                "int8 tensor value has no symmetric-quantization "
                "counterpart: -128"
            );
        }
    }
}

}  // namespace

Int8Tensor::Int8Tensor(Shape shape, std::vector<std::int8_t> values)
    : shape_m(std::move(shape)),
      data_m(std::move(values)) {
    if (data_m.size() != checked_numel(shape_m)) {
        throw std::invalid_argument(
            "int8 tensor data size does not match its shape"
        );
    }

    require_symmetric_range(data_m);
}

std::size_t Int8Tensor::rank() const {
    return shape_m.size();
}

std::size_t Int8Tensor::numel() const {
    return data_m.size();
}

const Int8Tensor::Shape& Int8Tensor::shape() const {
    return shape_m;
}

std::int8_t* Int8Tensor::data() {
    return data_m.data();
}

const std::int8_t* Int8Tensor::data() const {
    return data_m.data();
}

}  // namespace gpt2
