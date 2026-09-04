// Drives the GPT2CPP tokenizer so that the Python parity test can
// compare it with a Hugging Face tokenizer.
//
// Usage:
//   gpt2_tokenizer_runner <vocab.json> <merges.txt> <input> <output>
//
// Each input line is one request:
//   "e <hex>"   encode the UTF-8 bytes given in lowercase hexadecimal
//   "d <ids>"   decode the space-separated token IDs
//
// The runner answers every request with one output line: "i" followed by
// the space-separated token IDs for an encode request, or "b" followed
// by the lowercase hexadecimal bytes for a decode request.

#include "gpt2/tokenizer.h"

#include <cstddef>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <ios>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using TokenId = gpt2::Gpt2Tokenizer::TokenId;

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

std::string encode_hex(std::string_view bytes) {
    constexpr std::string_view digits = "0123456789abcdef";

    std::string text;
    text.reserve(bytes.size() * 2);
    for (const char character : bytes) {
        const auto byte = static_cast<unsigned char>(character);
        text.push_back(digits[byte >> 4U]);
        text.push_back(digits[byte & 0x0FU]);
    }

    return text;
}

std::vector<TokenId> parse_token_ids(std::string_view text) {
    std::istringstream stream{std::string(text)};
    std::vector<TokenId> token_ids;
    TokenId token_id = 0;

    while (stream >> token_id) {
        token_ids.push_back(token_id);
    }

    if (!stream.eof()) {
        throw std::runtime_error("request contains an invalid token ID");
    }

    return token_ids;
}

void answer_request(
    const gpt2::Gpt2Tokenizer& tokenizer,
    std::string_view request,
    std::ostream& output
) {
    if (request.size() < 1) {
        throw std::runtime_error("request is empty");
    }

    const char kind = request[0];
    std::string_view payload = request.substr(1);
    if (!payload.empty()) {
        if (payload.front() != ' ') {
            throw std::runtime_error("request has no separator");
        }
        payload.remove_prefix(1);
    }

    if (kind == 'e') {
        const std::vector<TokenId> token_ids =
            tokenizer.encode(decode_hex(payload));

        output << 'i';
        for (const TokenId token_id : token_ids) {
            output << ' ' << token_id;
        }
        output << '\n';
        return;
    }

    if (kind == 'd') {
        const std::vector<TokenId> token_ids = parse_token_ids(payload);
        output << 'b' << ' '
               << encode_hex(tokenizer.decode(token_ids)) << '\n';
        return;
    }

    throw std::runtime_error(
        std::string("request has an unknown kind: ") + kind
    );
}

}  // namespace

int main(int argument_count, char** arguments) {
    if (argument_count != 5) {
        std::cerr << "usage: gpt2_tokenizer_runner <vocab.json> "
                     "<merges.txt> <input> <output>\n";
        return EXIT_FAILURE;
    }

    try {
        const gpt2::Gpt2Tokenizer tokenizer = gpt2::Gpt2Tokenizer::load(
            arguments[1],
            arguments[2]
        );

        std::ifstream input(arguments[3], std::ios::binary);
        if (!input) {
            throw std::runtime_error(
                std::string("could not open the request file: ") +
                arguments[3]
            );
        }

        std::ofstream output(arguments[4], std::ios::binary);
        if (!output) {
            throw std::runtime_error(
                std::string("could not open the answer file: ") +
                arguments[4]
            );
        }

        std::string line;
        while (std::getline(input, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (line.empty()) {
                continue;
            }

            answer_request(tokenizer, line, output);
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

        std::cout << tokenizer.vocabulary_size() << '\n';
    } catch (const std::exception& exception) {
        std::cerr << "gpt2_tokenizer_runner failed: "
                  << exception.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
