#include "cli.h"

#include "gpt2/checkpoint.h"
#include "gpt2/generation.h"
#include "gpt2/tokenizer.h"

#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <locale>
#include <optional>
#include <ostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace gpt2 {

namespace {

constexpr std::size_t default_maximum_new_tokens = 32;
constexpr std::uint64_t default_seed = 42;

struct CliOptions {
    std::filesystem::path checkpoint;
    std::filesystem::path vocabulary;
    std::filesystem::path merges;
    std::string prompt;
    std::size_t maximum_new_tokens = default_maximum_new_tokens;
    SamplingOptions sampling;
    std::uint64_t seed = default_seed;
    bool sample = false;
    bool use_cache = true;
};

void print_help(std::ostream& output) {
    output <<
        "Usage:\n"
        "  gpt2 --checkpoint PATH --vocab PATH --merges PATH "
        "--prompt TEXT [options]\n\n"
        "Required:\n"
        "  --checkpoint PATH       Exported GPT-2 checkpoint\n"
        "  --vocab PATH            GPT-2 vocab.json\n"
        "  --merges PATH           GPT-2 merges.txt\n"
        "  --prompt TEXT           Text to continue\n\n"
        "Generation:\n"
        "  --max-new-tokens N      Token limit (default: 32)\n"
        "  --sample                Sample instead of greedy decoding\n"
        "  --temperature VALUE     Sampling temperature (default: 1)\n"
        "  --top-k N               Sampling top-k filter (default: 0)\n"
        "  --top-p VALUE           Sampling top-p filter (default: 1)\n"
        "  --seed N                Sampling seed (default: 42)\n"
        "  --no-cache              Disable the key/value cache\n\n"
        "Other:\n"
        "  -h, --help              Show this help\n";
}

bool requests_help(int argument_count, char* arguments[]) {
    for (int index = 1; index < argument_count; ++index) {
        const std::string_view argument(arguments[index]);
        if (argument == "-h" || argument == "--help") {
            return true;
        }
    }
    return false;
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

template <typename Integer>
Integer parse_integer(std::string_view text, std::string_view option) {
    Integer value = 0;
    const char* const end = text.data() + text.size();
    const auto result = std::from_chars(text.data(), end, value);
    if (text.empty() || result.ec != std::errc{} || result.ptr != end) {
        throw std::invalid_argument(
            std::string(option) + " requires a non-negative integer"
        );
    }
    return value;
}

float parse_float(std::string_view text, std::string_view option) {
    std::istringstream stream{std::string(text)};
    stream.imbue(std::locale::classic());

    float value = 0.0F;
    stream >> std::noskipws >> value;
    if (text.empty() || !stream ||
        stream.peek() != std::char_traits<char>::eof() ||
        !std::isfinite(value)) {
        throw std::invalid_argument(
            std::string(option) + " requires a finite number"
        );
    }
    return value;
}

template <typename Value>
void set_once(
    std::optional<Value>& destination,
    Value value,
    std::string_view option
) {
    if (destination.has_value()) {
        throw std::invalid_argument(
            std::string(option) + " may only be provided once"
        );
    }
    destination = std::move(value);
}

CliOptions parse_options(int argument_count, char* arguments[]) {
    std::optional<std::filesystem::path> checkpoint;
    std::optional<std::filesystem::path> vocabulary;
    std::optional<std::filesystem::path> merges;
    std::optional<std::string> prompt;
    std::optional<std::size_t> maximum_new_tokens;
    std::optional<float> temperature;
    std::optional<std::size_t> top_k;
    std::optional<float> top_p;
    std::optional<std::uint64_t> seed;
    bool sample = false;
    bool sample_seen = false;
    bool use_cache = true;
    bool cache_option_seen = false;

    for (int index = 1; index < argument_count; ++index) {
        const std::string_view argument(arguments[index]);

        if (argument == "--checkpoint") {
            set_once(
                checkpoint,
                std::filesystem::path(std::string(option_value(
                    index, argument_count, arguments, argument
                ))),
                argument
            );
        } else if (argument == "--vocab") {
            set_once(
                vocabulary,
                std::filesystem::path(std::string(option_value(
                    index, argument_count, arguments, argument
                ))),
                argument
            );
        } else if (argument == "--merges") {
            set_once(
                merges,
                std::filesystem::path(std::string(option_value(
                    index, argument_count, arguments, argument
                ))),
                argument
            );
        } else if (argument == "--prompt") {
            set_once(
                prompt,
                std::string(option_value(
                    index, argument_count, arguments, argument
                )),
                argument
            );
        } else if (argument == "--max-new-tokens") {
            set_once(
                maximum_new_tokens,
                parse_integer<std::size_t>(
                    option_value(index, argument_count, arguments, argument),
                    argument
                ),
                argument
            );
        } else if (argument == "--temperature") {
            set_once(
                temperature,
                parse_float(
                    option_value(index, argument_count, arguments, argument),
                    argument
                ),
                argument
            );
        } else if (argument == "--top-k") {
            set_once(
                top_k,
                parse_integer<std::size_t>(
                    option_value(index, argument_count, arguments, argument),
                    argument
                ),
                argument
            );
        } else if (argument == "--top-p") {
            set_once(
                top_p,
                parse_float(
                    option_value(index, argument_count, arguments, argument),
                    argument
                ),
                argument
            );
        } else if (argument == "--seed") {
            set_once(
                seed,
                parse_integer<std::uint64_t>(
                    option_value(index, argument_count, arguments, argument),
                    argument
                ),
                argument
            );
        } else if (argument == "--sample") {
            if (sample_seen) {
                throw std::invalid_argument(
                    "--sample may only be provided once"
                );
            }
            sample_seen = true;
            sample = true;
        } else if (argument == "--no-cache") {
            if (cache_option_seen) {
                throw std::invalid_argument(
                    "--no-cache may only be provided once"
                );
            }
            cache_option_seen = true;
            use_cache = false;
        } else {
            throw std::invalid_argument(
                "unknown option: " + std::string(argument)
            );
        }
    }

    if (!checkpoint.has_value()) {
        throw std::invalid_argument("missing required option --checkpoint");
    }
    if (!vocabulary.has_value()) {
        throw std::invalid_argument("missing required option --vocab");
    }
    if (!merges.has_value()) {
        throw std::invalid_argument("missing required option --merges");
    }
    if (!prompt.has_value()) {
        throw std::invalid_argument("missing required option --prompt");
    }
    if (prompt->empty()) {
        throw std::invalid_argument("--prompt must not be empty");
    }

    if (!sample && (temperature.has_value() || top_k.has_value() ||
                    top_p.has_value() || seed.has_value())) {
        throw std::invalid_argument(
            "sampling options require --sample"
        );
    }

    CliOptions options;
    options.checkpoint = std::move(*checkpoint);
    options.vocabulary = std::move(*vocabulary);
    options.merges = std::move(*merges);
    options.prompt = std::move(*prompt);
    options.sample = sample;
    options.use_cache = use_cache;

    if (maximum_new_tokens.has_value()) {
        options.maximum_new_tokens = *maximum_new_tokens;
    }
    if (temperature.has_value()) {
        options.sampling.temperature = *temperature;
    }
    if (top_k.has_value()) {
        options.sampling.top_k = *top_k;
    }
    if (top_p.has_value()) {
        options.sampling.top_p = *top_p;
    }
    if (seed.has_value()) {
        options.seed = *seed;
    }

    if (options.sampling.temperature <= 0.0F) {
        throw std::invalid_argument(
            "--temperature must be greater than zero"
        );
    }
    if (options.sampling.top_p <= 0.0F ||
        options.sampling.top_p > 1.0F) {
        throw std::invalid_argument(
            "--top-p must be greater than zero and at most one"
        );
    }

    return options;
}

}  // namespace

int run_cli(
    int argument_count,
    char* arguments[],
    std::ostream& output,
    std::ostream& error
) {
    if (requests_help(argument_count, arguments)) {
        print_help(output);
        return EXIT_SUCCESS;
    }

    try {
        const CliOptions options = parse_options(argument_count, arguments);
        const Gpt2Tokenizer tokenizer = Gpt2Tokenizer::load(
            options.vocabulary,
            options.merges
        );
        const Gpt2Model model(load_checkpoint(options.checkpoint));

        if (tokenizer.vocabulary_size() !=
            static_cast<std::size_t>(model.config().vocab_size)) {
            throw std::runtime_error(
                "tokenizer vocabulary size does not match the checkpoint"
            );
        }

        const std::vector<std::size_t> prompt_tokens =
            tokenizer.encode(options.prompt);

        GenerationLimits limits;
        limits.maximum_new_tokens = options.maximum_new_tokens;
        limits.end_of_text_id = tokenizer.end_of_text_id();
        limits.use_cache = options.use_cache;

        Generation generation;
        if (options.sample) {
            std::mt19937_64 generator(options.seed);
            generation = generate_sampled(
                model,
                prompt_tokens,
                limits,
                options.sampling,
                generator
            );
        } else {
            generation = generate_greedy(model, prompt_tokens, limits);
        }

        output << options.prompt
               << tokenizer.decode(generation.new_token_ids)
               << '\n';
    } catch (const std::exception& exception) {
        error << "error: " << exception.what() << '\n'
              << "Run 'gpt2 --help' for usage.\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

}  // namespace gpt2
