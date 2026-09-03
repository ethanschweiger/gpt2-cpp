#pragma once

#include "gpt2/tensor.h"

#include <cstddef>
#include <span>

namespace gpt2 {

Tensor embedding_lookup(
    const Tensor& embedding_table,
    std::span<const std::size_t> token_ids
);

}  // namespace gpt2
