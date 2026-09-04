// Applies the sampling filters to score vectors so that the Python
// parity test can compare them with Hugging Face's logits warpers.
//
// Usage:
//   gpt2_sampling_runner <cases> <scores.bin> <probabilities.bin>
//
// Each line of <cases> is "<count> <temperature> <top_k> <top_p>". The
// runner reads that many little-endian float32 scores from <scores.bin>
// in order, and appends the same number of float32 probabilities to
// <probabilities.bin>.

#include "gpt2/generation.h"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <ios>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::uint32_t read_u32(std::istream& input) {
    std::array<unsigned char, 4> bytes{};
    input.read(
        reinterpret_cast<char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size())
    );
    if (!input) {
        throw std::runtime_error("score file ended early");
    }

    return static_cast<std::uint32_t>(bytes[0]) |
        (static_cast<std::uint32_t>(bytes[1]) << 8U) |
        (static_cast<std::uint32_t>(bytes[2]) << 16U) |
        (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

void write_u32(std::ostream& output, std::uint32_t value) {
    for (unsigned int shift = 0; shift < 32; shift += 8) {
        const auto byte = static_cast<unsigned char>(value >> shift);
        output.put(static_cast<char>(byte));
    }
}

}  // namespace

int main(int argument_count, char** arguments) {
    if (argument_count != 4) {
        std::cerr << "usage: gpt2_sampling_runner <cases> <scores.bin> "
                     "<probabilities.bin>\n";
        return EXIT_FAILURE;
    }

    try {
        std::ifstream cases(arguments[1]);
        if (!cases) {
            throw std::runtime_error(
                std::string("could not open the case file: ") +
                arguments[1]
            );
        }

        std::ifstream scores(arguments[2], std::ios::binary);
        if (!scores) {
            throw std::runtime_error(
                std::string("could not open the score file: ") +
                arguments[2]
            );
        }

        std::ofstream output(arguments[3], std::ios::binary);
        if (!output) {
            throw std::runtime_error(
                std::string("could not open the output file: ") +
                arguments[3]
            );
        }

        std::string line;
        std::size_t case_count = 0;
        while (std::getline(cases, line)) {
            if (line.empty()) {
                continue;
            }

            std::istringstream fields(line);
            std::size_t count = 0;
            gpt2::SamplingOptions options;
            fields >> count >> options.temperature >> options.top_k >>
                options.top_p;
            if (!fields) {
                throw std::runtime_error("case line is malformed: " + line);
            }
            if (count == 0) {
                throw std::runtime_error("case has no scores");
            }

            std::vector<float> values(count);
            for (float& value : values) {
                value = std::bit_cast<float>(read_u32(scores));
            }

            const std::vector<float> probabilities =
                gpt2::sampling_distribution(values, options);
            if (probabilities.size() != count) {
                throw std::runtime_error(
                    "sampling returned the wrong number of probabilities"
                );
            }

            for (const float probability : probabilities) {
                write_u32(
                    output,
                    std::bit_cast<std::uint32_t>(probability)
                );
            }
            ++case_count;
        }

        if (cases.bad() || scores.bad()) {
            throw std::runtime_error("I/O error while reading input");
        }

        output.flush();
        if (!output) {
            throw std::runtime_error("I/O error while writing output");
        }

        std::cout << case_count << '\n';
    } catch (const std::exception& exception) {
        std::cerr << "gpt2_sampling_runner failed: "
                  << exception.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
