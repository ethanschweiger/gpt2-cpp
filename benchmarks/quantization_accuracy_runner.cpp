// Runs a real checkpoint over a fixed evaluation text and writes its
// full-sequence logits (plus the token IDs the text encoded to) so a
// driver script can compare configurations directly against each
// other, matching Gpt2Model::forward()'s [sequence length, vocabulary
// size] output. Unlike tests/real_model_runner.cpp (a handful of
// hand-picked token IDs, compared against Hugging Face), this exists
// to compare gpt2-cpp checkpoints against *each other* -- the FP32
// baseline against its two quantized siblings -- over real, coherent
// English text rather than a handful of synthetic token IDs, since a
// language model's activation statistics on real text are not what
// they are on arbitrary token sequences. See docs/quantization.md.

#include "gpt2/checkpoint.h"
#include "gpt2/model.h"
#include "gpt2/tokenizer.h"

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Options {
    std::filesystem::path checkpoint;
    std::filesystem::path vocabulary;
    std::filesystem::path merges;
    std::filesystem::path text;
    std::filesystem::path output;
};

void print_help(std::ostream& output) {
    output <<
        "Usage:\n"
        "  gpt2_quantization_accuracy_runner --checkpoint PATH "
        "--vocab PATH --merges PATH --text PATH --output PATH\n\n"
        "Encodes the text file with GPT-2's tokenizer, runs it through\n"
        "the checkpoint in one forward pass, and writes the resulting\n"
        "[sequence length, vocabulary size] logits as raw little-endian\n"
        "float32 to the output path. Prints \"ROWS COLS\" and then the\n"
        "encoded token IDs (space-separated) to stdout.\n";
}

std::string_view option_value(
    int& index,
    int argument_count,
    char* arguments[],
    std::string_view option
) {
    if (index + 1 >= argument_count) {
        throw std::invalid_argument(
            std::string(option) + " requires a value"
        );
    }
    ++index;
    return arguments[index];
}

Options parse_options(int argument_count, char* arguments[]) {
    std::optional<std::filesystem::path> checkpoint;
    std::optional<std::filesystem::path> vocabulary;
    std::optional<std::filesystem::path> merges;
    std::optional<std::filesystem::path> text;
    std::optional<std::filesystem::path> output;

    for (int index = 1; index < argument_count; ++index) {
        const std::string_view argument(arguments[index]);
        if (argument == "--checkpoint") {
            checkpoint = std::filesystem::path(std::string(
                option_value(index, argument_count, arguments, argument)
            ));
        } else if (argument == "--vocab") {
            vocabulary = std::filesystem::path(std::string(
                option_value(index, argument_count, arguments, argument)
            ));
        } else if (argument == "--merges") {
            merges = std::filesystem::path(std::string(
                option_value(index, argument_count, arguments, argument)
            ));
        } else if (argument == "--text") {
            text = std::filesystem::path(std::string(
                option_value(index, argument_count, arguments, argument)
            ));
        } else if (argument == "--output") {
            output = std::filesystem::path(std::string(
                option_value(index, argument_count, arguments, argument)
            ));
        } else {
            throw std::invalid_argument(
                "unknown option: " + std::string(argument)
            );
        }
    }

    if (!checkpoint.has_value() || !vocabulary.has_value() ||
        !merges.has_value() || !text.has_value() ||
        !output.has_value()) {
        throw std::invalid_argument(
            "--checkpoint, --vocab, --merges, --text and --output are "
            "all required"
        );
    }

    return Options{
        std::move(*checkpoint),
        std::move(*vocabulary),
        std::move(*merges),
        std::move(*text),
        std::move(*output),
    };
}

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error(
            "could not open evaluation text: " + path.string()
        );
    }

    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

void write_logits(
    const gpt2::Tensor& logits,
    const std::filesystem::path& output_path
) {
    const std::size_t maximum_value_count =
        static_cast<std::size_t>(
            std::numeric_limits<std::streamsize>::max()
        ) / sizeof(float);
    if (logits.numel() > maximum_value_count) {
        throw std::runtime_error("logit output is too large to write");
    }

    const std::size_t byte_count = logits.numel() * sizeof(float);
    std::ofstream output(output_path, std::ios::binary);
    output.write(
        reinterpret_cast<const char*>(logits.data()),
        static_cast<std::streamsize>(byte_count)
    );
    output.close();

    if (!output) {
        throw std::runtime_error(
            "could not write logits: " + output_path.string()
        );
    }
}

}  // namespace

int main(int argument_count, char* arguments[]) {
    for (int index = 1; index < argument_count; ++index) {
        const std::string_view argument(arguments[index]);
        if (argument == "-h" || argument == "--help") {
            print_help(std::cout);
            return EXIT_SUCCESS;
        }
    }

    try {
        const Options options = parse_options(argument_count, arguments);

        const gpt2::Gpt2Tokenizer tokenizer = gpt2::Gpt2Tokenizer::load(
            options.vocabulary,
            options.merges
        );
        const gpt2::Gpt2Model model(
            gpt2::load_checkpoint(options.checkpoint)
        );

        if (tokenizer.vocabulary_size() !=
            static_cast<std::size_t>(model.config().vocab_size)) {
            throw std::runtime_error(
                "tokenizer vocabulary size does not match the checkpoint"
            );
        }

        const std::vector<std::size_t> token_ids =
            tokenizer.encode(read_text_file(options.text));

        const std::size_t context_length =
            static_cast<std::size_t>(model.config().context_length);
        if (token_ids.empty()) {
            throw std::runtime_error(
                "evaluation text encoded to zero tokens"
            );
        }
        if (token_ids.size() > context_length) {
            throw std::runtime_error(
                "evaluation text (" + std::to_string(token_ids.size()) +
                " tokens) exceeds the checkpoint context length (" +
                std::to_string(context_length) + ")"
            );
        }

        const gpt2::Tensor logits = model.forward(token_ids);
        write_logits(logits, options.output);

        std::cout << logits.shape()[0] << ' ' << logits.shape()[1]
                   << '\n';
        for (std::size_t index = 0; index < token_ids.size(); ++index) {
            if (index != 0) {
                std::cout << ' ';
            }
            std::cout << token_ids[index];
        }
        std::cout << '\n';
    } catch (const std::exception& exception) {
        std::cerr << "Quantization accuracy runner failed: "
                   << exception.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
