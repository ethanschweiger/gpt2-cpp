#include "gpt2/layers.h"

#include <algorithm>
#include <stdexcept>

namespace gpt2 {

Tensor embedding_lookup(
    const Tensor& embedding_table,
    std::span<const std::size_t> token_ids
) {
    if (embedding_table.rank() != 2) {
        throw std::invalid_argument(
            "embedding table must be a rank-2 tensor"
        );
    }

    if (token_ids.empty()) {
        throw std::invalid_argument(
            "embedding lookup requires at least one token"
        );
    }

    const std::size_t vocabulary_size =
        embedding_table.shape()[0];
    const std::size_t embedding_size =
        embedding_table.shape()[1];

    Tensor result({token_ids.size(), embedding_size});

    const float* table_data = embedding_table.data();
    float* result_data = result.data();

    for (std::size_t token_index = 0;
         token_index < token_ids.size();
         ++token_index) {
        const std::size_t token_id = token_ids[token_index];

        if (token_id >= vocabulary_size) {
            throw std::out_of_range(
                "token ID is outside the embedding vocabulary"
            );
        }

        const float* source =
            table_data + token_id * embedding_size;
        float* destination =
            result_data + token_index * embedding_size;

        std::copy_n(source, embedding_size, destination);
    }

    return result;
}

}  // namespace gpt2
