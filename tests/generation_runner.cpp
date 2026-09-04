// Drives end-to-end greedy generation so that the Python parity test
// can compare GPT2CPP with Hugging Face.
//
// Usage:
//   gpt2_generation_runner <checkpoint> <vocab.json> <merges.txt>
//                          <requests> <answers>
//
// Each request line is "<max new tokens> <prompt hex>", where the hex
// is the lowercase encoding of the prompt's UTF-8 bytes. Each answer
// line is "<stop reason> <space-separated generated token IDs>".

#include "gpt2/checkpoint.h"
#include "gpt2/generation.h"
#include "gpt2/tokenizer.h"

#include <charconv>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <ios>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

unsigned char parse_hex_digit(char digit) {
    if (digit >= '0' && digit <= '9') {
        return static_cast<unsigned char>(digit - '0');
    }
    if (digit >= 'a' && digit <= 'f') {
        return static_cast<unsigned char>(digit - 'a' + 10);
    }

    throw std::runtime_error(
        std::string("request contains a non-hexadecimal digit: ") + digit
    );
}

std::string decode_hex(std::string_view text) {
    if (text.size() % 2 != 0) {
        throw std::runtime_error(
            "request has an odd number of hexadecimal digits"
        );
    }

    std::string bytes;
    bytes.reserve(text.size() / 2);
    for (std::size_t index = 0; index < text.size(); index += 2) {
        const auto high = parse_hex_digit(text[index]);
        const auto low = parse_hex_digit(text[index + 1]);
        bytes.push_back(static_cast<char>(
            static_cast<unsigned char>((high << 4U) | low)
        ));
    }

    return bytes;
}

std::size_t parse_count(std::string_view text) {
    std::size_t value = 0;
    const char* const end = text.data() + text.size();
    const auto result = std::from_chars(text.data(), end, value);
    if (result.ec != std::errc{} || result.ptr != end) {
        throw std::runtime_error(
            "request has an invalid token count: " + std::string(text)
        );
    }

    return value;
}

std::string_view describe(gpt2::GenerationStop stop) {
    switch (stop) {
    case gpt2::GenerationStop::token_limit:
        return "token_limit";
    case gpt2::GenerationStop::end_of_text:
        return "end_of_text";
    case gpt2::GenerationStop::context_limit:
        return "context_limit";
    }

    throw std::logic_error("unknown generation stop reason");
}

}  // namespace

int main(int argument_count, char** arguments) {
    if (argument_count != 6) {
        std::cerr << "usage: gpt2_generation_runner <checkpoint> "
                     "<vocab.json> <merges.txt> <requests> <answers>\n";
        return EXIT_FAILURE;
    }

    try {
        const gpt2::Gpt2Tokenizer tokenizer = gpt2::Gpt2Tokenizer::load(
            arguments[2],
            arguments[3]
        );
        const gpt2::Gpt2Model model(
            gpt2::load_checkpoint(arguments[1])
        );

        std::ifstream input(arguments[4], std::ios::binary);
        if (!input) {
            throw std::runtime_error(
                std::string("could not open the request file: ") +
                arguments[4]
            );
        }

        std::ofstream output(arguments[5], std::ios::binary);
        if (!output) {
            throw std::runtime_error(
                std::string("could not open the answer file: ") +
                arguments[5]
            );
        }

        gpt2::GreedyGenerationOptions options;
        options.end_of_text_id = tokenizer.end_of_text_id();

        std::string line;
        while (std::getline(input, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (line.empty()) {
                continue;
            }

            const std::size_t separator = line.find(' ');
            if (separator == std::string::npos) {
                throw std::runtime_error("request has no separator");
            }

            options.maximum_new_tokens =
                parse_count(std::string_view(line).substr(0, separator));
            const std::string prompt = decode_hex(
                std::string_view(line).substr(separator + 1)
            );

            const std::vector<std::size_t> prompt_token_ids =
                tokenizer.encode(prompt);
            const gpt2::GreedyGeneration generation =
                gpt2::generate_greedy(model, prompt_token_ids, options);

            output << describe(generation.stop);
            for (const std::size_t token_id : generation.new_token_ids) {
                output << ' ' << token_id;
            }
            output << '\n';
        }

        if (input.bad()) {
            throw std::runtime_error(
                "I/O error while reading the request file"
            );
        }

        output.flush();
        if (!output) {
            throw std::runtime_error(
                "I/O error while writing the answer file"
            );
        }

        std::cout << model.config().context_length << '\n';
    } catch (const std::exception& exception) {
        std::cerr << "gpt2_generation_runner failed: "
                  << exception.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
