#include "gpt2/checkpoint.h"
#include "gpt2/model.h"

#include <charconv>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

std::size_t parse_token_id(std::string_view text) {
    std::size_t token_id = 0;
    const char* const end = text.data() + text.size();
    const auto result = std::from_chars(
        text.data(),
        end,
        token_id
    );

    if (result.ec != std::errc{} || result.ptr != end) {
        throw std::invalid_argument(
            "invalid token ID: " + std::string(text)
        );
    }

    return token_id;
}

void write_logits(
    const gpt2::Tensor& logits,
    const char* output_path
) {
    const std::size_t maximum_value_count =
        static_cast<std::size_t>(
            std::numeric_limits<std::streamsize>::max()
        ) / sizeof(float);
    if (logits.numel() > maximum_value_count) {
        throw std::runtime_error(
            "logit output is too large to write"
        );
    }

    const std::size_t byte_count =
        logits.numel() * sizeof(float);
    std::ofstream output(output_path, std::ios::binary);
    output.write(
        reinterpret_cast<const char*>(logits.data()),
        static_cast<std::streamsize>(byte_count)
    );
    output.close();

    if (!output) {
        throw std::runtime_error(
            "could not write reference-test logits"
        );
    }
}

}  // namespace

int main(int argument_count, char* arguments[]) {
    if (argument_count < 4) {
        std::cerr
            << "Usage: gpt2_real_model_runner "
            << "CHECKPOINT OUTPUT TOKEN_ID...\n";
        return EXIT_FAILURE;
    }

    try {
        std::vector<std::size_t> token_ids;
        token_ids.reserve(
            static_cast<std::size_t>(argument_count - 3)
        );
        for (int argument = 3;
             argument < argument_count;
             ++argument) {
            token_ids.push_back(parse_token_id(arguments[argument]));
        }

        const gpt2::Gpt2Model model(
            gpt2::load_checkpoint(arguments[1])
        );
        const gpt2::Tensor logits =
            model.forward(token_ids);

        write_logits(logits, arguments[2]);

        std::cout << logits.shape()[0]
                  << ' '
                  << logits.shape()[1]
                  << '\n';
    } catch (const std::exception& exception) {
        std::cerr << "Real-model runner failed: "
                  << exception.what()
                  << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
