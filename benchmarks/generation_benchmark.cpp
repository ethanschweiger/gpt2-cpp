#include "gpt2/checkpoint.h"
#include "gpt2/generation.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#ifndef GPT2_BENCHMARK_BUILD_TYPE
#define GPT2_BENCHMARK_BUILD_TYPE "unknown"
#endif

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t default_prompt_tokens = 8;
constexpr std::size_t default_new_tokens = 24;
constexpr std::size_t default_warmups = 1;
constexpr std::size_t default_trials = 3;

struct Options {
    std::filesystem::path checkpoint;
    std::optional<std::filesystem::path> json_output;
    std::size_t prompt_tokens = default_prompt_tokens;
    std::size_t new_tokens = default_new_tokens;
    std::size_t warmups = default_warmups;
    std::size_t trials = default_trials;
};

struct TrialPair {
    double cached_seconds;
    double uncached_seconds;
};

struct Results {
    gpt2::ModelConfig config;
    std::uintmax_t checkpoint_bytes;
    std::vector<std::size_t> prompt_token_ids;
    double checkpoint_load_seconds;
    std::vector<double> cached_trials;
    std::vector<double> uncached_trials;
    double cached_median;
    double uncached_median;
};

void print_help(std::ostream& output) {
    output <<
        "Usage:\n"
        "  gpt2_generation_benchmark --checkpoint PATH [options]\n\n"
        "Options:\n"
        "  --prompt-tokens N   Synthetic prompt length (default: 8)\n"
        "  --new-tokens N      Generated token count (default: 24)\n"
        "  --warmups N         Unmeasured trial pairs (default: 1)\n"
        "  --trials N          Measured trial pairs (default: 3)\n"
        "  --json PATH         Also write machine-readable results\n"
        "  -h, --help          Show this help\n";
}

std::string_view require_value(
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

std::size_t parse_count(std::string_view text, std::string_view option) {
    std::size_t value = 0;
    const char* const end = text.data() + text.size();
    const auto result = std::from_chars(text.data(), end, value);
    if (text.empty() || result.ec != std::errc{} || result.ptr != end) {
        throw std::invalid_argument(
            std::string(option) + " requires a non-negative integer"
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

Options parse_options(int argument_count, char* arguments[]) {
    std::optional<std::filesystem::path> checkpoint;
    std::optional<std::filesystem::path> json_output;
    std::optional<std::size_t> prompt_tokens;
    std::optional<std::size_t> new_tokens;
    std::optional<std::size_t> warmups;
    std::optional<std::size_t> trials;

    for (int index = 1; index < argument_count; ++index) {
        const std::string_view argument(arguments[index]);
        if (argument == "--checkpoint") {
            set_once(
                checkpoint,
                std::filesystem::path(std::string(require_value(
                    index, argument_count, arguments, argument
                ))),
                argument
            );
        } else if (argument == "--json") {
            set_once(
                json_output,
                std::filesystem::path(std::string(require_value(
                    index, argument_count, arguments, argument
                ))),
                argument
            );
        } else if (argument == "--prompt-tokens") {
            set_once(
                prompt_tokens,
                parse_count(
                    require_value(index, argument_count, arguments, argument),
                    argument
                ),
                argument
            );
        } else if (argument == "--new-tokens") {
            set_once(
                new_tokens,
                parse_count(
                    require_value(index, argument_count, arguments, argument),
                    argument
                ),
                argument
            );
        } else if (argument == "--warmups") {
            set_once(
                warmups,
                parse_count(
                    require_value(index, argument_count, arguments, argument),
                    argument
                ),
                argument
            );
        } else if (argument == "--trials") {
            set_once(
                trials,
                parse_count(
                    require_value(index, argument_count, arguments, argument),
                    argument
                ),
                argument
            );
        } else {
            throw std::invalid_argument(
                "unknown option: " + std::string(argument)
            );
        }
    }

    if (!checkpoint.has_value()) {
        throw std::invalid_argument("missing required option --checkpoint");
    }

    Options options;
    options.checkpoint = std::move(*checkpoint);
    options.json_output = std::move(json_output);
    if (prompt_tokens.has_value()) {
        options.prompt_tokens = *prompt_tokens;
    }
    if (new_tokens.has_value()) {
        options.new_tokens = *new_tokens;
    }
    if (warmups.has_value()) {
        options.warmups = *warmups;
    }
    if (trials.has_value()) {
        options.trials = *trials;
    }

    if (options.prompt_tokens == 0) {
        throw std::invalid_argument("--prompt-tokens must be positive");
    }
    if (options.new_tokens == 0) {
        throw std::invalid_argument("--new-tokens must be positive");
    }
    if (options.trials == 0) {
        throw std::invalid_argument("--trials must be positive");
    }

    return options;
}

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

double time_generation(
    const gpt2::Gpt2Model& model,
    const std::vector<std::size_t>& prompt,
    std::size_t new_tokens,
    bool use_cache,
    gpt2::Generation& generation
) {
    gpt2::GenerationLimits limits;
    limits.maximum_new_tokens = new_tokens;
    limits.use_cache = use_cache;

    const Clock::time_point start = Clock::now();
    generation = gpt2::generate_greedy(model, prompt, limits);
    return seconds_since(start);
}

void verify_pair(
    const gpt2::Generation& cached,
    const gpt2::Generation& uncached,
    std::size_t expected_tokens
) {
    if (cached.new_token_ids.size() != expected_tokens ||
        uncached.new_token_ids.size() != expected_tokens) {
        throw std::runtime_error(
            "a benchmark run stopped before generating every requested token"
        );
    }
    if (cached.new_token_ids != uncached.new_token_ids ||
        cached.stop != uncached.stop) {
        throw std::runtime_error(
            "cached and uncached generation produced different results"
        );
    }
}

TrialPair run_pair(
    const gpt2::Gpt2Model& model,
    const std::vector<std::size_t>& prompt,
    std::size_t new_tokens,
    bool cached_first
) {
    gpt2::Generation cached;
    gpt2::Generation uncached;
    double cached_seconds = 0.0;
    double uncached_seconds = 0.0;

    if (cached_first) {
        cached_seconds = time_generation(
            model, prompt, new_tokens, true, cached
        );
        uncached_seconds = time_generation(
            model, prompt, new_tokens, false, uncached
        );
    } else {
        uncached_seconds = time_generation(
            model, prompt, new_tokens, false, uncached
        );
        cached_seconds = time_generation(
            model, prompt, new_tokens, true, cached
        );
    }

    verify_pair(cached, uncached, new_tokens);
    return TrialPair{cached_seconds, uncached_seconds};
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 != 0) {
        return values[middle];
    }
    return (values[middle - 1] + values[middle]) / 2.0;
}

std::string compiler_description() {
#if defined(__clang__)
    return std::string("Clang ") + __clang_version__;
#elif defined(__GNUC__)
    return std::string("GCC ") + __VERSION__;
#elif defined(_MSC_VER)
    return "MSVC " + std::to_string(_MSC_VER);
#else
    return "unknown";
#endif
}

std::string_view platform_name() {
#if defined(__APPLE__)
    return "macOS";
#elif defined(__linux__)
    return "Linux";
#elif defined(_WIN32)
    return "Windows";
#else
    return "unknown";
#endif
}

std::string_view architecture_name() {
#if defined(__aarch64__) || defined(_M_ARM64)
    return "arm64";
#elif defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#else
    return "unknown";
#endif
}

std::string json_escape(std::string_view text) {
    std::string escaped;
    for (const char character : text) {
        switch (character) {
        case '\\':
            escaped += "\\\\";
            break;
        case '"':
            escaped += "\\\"";
            break;
        case '\n':
            escaped += "\\n";
            break;
        case '\r':
            escaped += "\\r";
            break;
        case '\t':
            escaped += "\\t";
            break;
        default:
            escaped.push_back(character);
            break;
        }
    }
    return escaped;
}

void write_number_array(
    std::ostream& output,
    const std::vector<double>& values
) {
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            output << ", ";
        }
        output << values[index];
    }
    output << ']';
}

void write_token_array(
    std::ostream& output,
    const std::vector<std::size_t>& values
) {
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            output << ", ";
        }
        output << values[index];
    }
    output << ']';
}

void write_json(
    std::ostream& output,
    const Options& options,
    const gpt2::ModelConfig& config,
    const Results& results
) {
    const double generated = static_cast<double>(options.new_tokens);
    output << std::setprecision(17)
           << "{\n"
           << "  \"schema_version\": 1,\n"
           << "  \"environment\": {\n"
           << "    \"platform\": \"" << platform_name() << "\",\n"
           << "    \"architecture\": \"" << architecture_name()
           << "\",\n"
           << "    \"compiler\": \""
           << json_escape(compiler_description()) << "\",\n"
           << "    \"build_type\": \""
           << json_escape(GPT2_BENCHMARK_BUILD_TYPE) << "\",\n"
           << "    \"cpp_standard\": " << __cplusplus << ",\n"
           << "    \"hardware_threads\": "
           << std::thread::hardware_concurrency() << "\n"
           << "  },\n"
           << "  \"model\": {\n"
           << "    \"checkpoint_bytes\": "
           << results.checkpoint_bytes << ",\n"
           << "    \"vocabulary_size\": " << config.vocab_size << ",\n"
           << "    \"context_length\": " << config.context_length << ",\n"
           << "    \"embedding_size\": " << config.embedding_size << ",\n"
           << "    \"head_count\": " << config.head_count << ",\n"
           << "    \"layer_count\": " << config.layer_count << "\n"
           << "  },\n"
           << "  \"benchmark\": {\n"
           << "    \"prompt_tokens\": " << options.prompt_tokens << ",\n"
           << "    \"prompt_token_ids\": ";
    write_token_array(output, results.prompt_token_ids);
    output << ",\n"
           << "    \"new_tokens\": " << options.new_tokens << ",\n"
           << "    \"warmups\": " << options.warmups << ",\n"
           << "    \"trials\": " << options.trials << "\n"
           << "  },\n"
           << "  \"results\": {\n"
           << "    \"checkpoint_load_seconds\": "
           << results.checkpoint_load_seconds << ",\n"
           << "    \"cached_trial_seconds\": ";
    write_number_array(output, results.cached_trials);
    output << ",\n    \"uncached_trial_seconds\": ";
    write_number_array(output, results.uncached_trials);
    output << ",\n"
           << "    \"cached_median_seconds\": "
           << results.cached_median << ",\n"
           << "    \"uncached_median_seconds\": "
           << results.uncached_median << ",\n"
           << "    \"cached_tokens_per_second\": "
           << generated / results.cached_median << ",\n"
           << "    \"uncached_tokens_per_second\": "
           << generated / results.uncached_median << ",\n"
           << "    \"cache_speedup\": "
           << results.uncached_median / results.cached_median << ",\n"
           << "    \"tokens_agree\": true\n"
           << "  }\n"
           << "}\n";
}

void print_results(
    std::ostream& output,
    const Options& options,
    const gpt2::ModelConfig& config,
    const Results& results
) {
    const double generated = static_cast<double>(options.new_tokens);
    output << "GPT2CPP generation benchmark\n"
           << "build: " << GPT2_BENCHMARK_BUILD_TYPE << '\n'
           << "compiler: " << compiler_description() << '\n'
           << "platform: " << platform_name() << ' ' << architecture_name()
           << '\n'
           << "hardware threads: " << std::thread::hardware_concurrency()
           << '\n'
           << "model: " << config.layer_count << " layers, "
           << config.embedding_size << " embedding, "
           << config.head_count << " heads\n"
           << "checkpoint bytes: " << results.checkpoint_bytes << '\n'
           << "prompt tokens: " << options.prompt_tokens << '\n'
           << "new tokens: " << options.new_tokens << '\n'
           << "warmup pairs: " << options.warmups << '\n'
           << "measured pairs: " << options.trials << '\n'
           << std::fixed << std::setprecision(3)
           << "checkpoint load: " << results.checkpoint_load_seconds
           << " s\n"
           << "cached median: " << results.cached_median << " s ("
           << generated / results.cached_median << " tokens/s)\n"
           << "uncached median: " << results.uncached_median << " s ("
           << generated / results.uncached_median << " tokens/s)\n"
           << "cache speedup: "
           << results.uncached_median / results.cached_median << "x\n"
           << "tokens agree: yes\n";
}

Results benchmark(const Options& options, std::ostream& progress) {
    const std::uintmax_t checkpoint_bytes =
        std::filesystem::file_size(options.checkpoint);
    progress << "Loading checkpoint..." << std::flush;
    const Clock::time_point load_start = Clock::now();
    const gpt2::Gpt2Model model(
        gpt2::load_checkpoint(options.checkpoint)
    );
    const double checkpoint_load_seconds = seconds_since(load_start);
    progress << " done (" << std::fixed << std::setprecision(3)
             << checkpoint_load_seconds << " s)\n";

    const std::size_t context_length =
        static_cast<std::size_t>(model.config().context_length);
    if (options.prompt_tokens >= context_length ||
        options.new_tokens > context_length - options.prompt_tokens) {
        throw std::invalid_argument(
            "prompt and generated tokens must fit inside the context window"
        );
    }

    const std::size_t vocabulary_size =
        static_cast<std::size_t>(model.config().vocab_size);
    std::vector<std::size_t> prompt(options.prompt_tokens);
    for (std::size_t index = 0; index < prompt.size(); ++index) {
        prompt[index] = (index * 37 + 100) % vocabulary_size;
    }

    for (std::size_t warmup = 0; warmup < options.warmups; ++warmup) {
        progress << "Warmup pair " << warmup + 1 << '/' << options.warmups
                 << "..." << std::flush;
        static_cast<void>(run_pair(
            model,
            prompt,
            options.new_tokens,
            warmup % 2 == 0
        ));
        progress << " done\n";
    }

    std::vector<double> cached_trials;
    std::vector<double> uncached_trials;
    cached_trials.reserve(options.trials);
    uncached_trials.reserve(options.trials);

    for (std::size_t trial = 0; trial < options.trials; ++trial) {
        progress << "Measured pair " << trial + 1 << '/' << options.trials
                 << "..." << std::flush;
        const TrialPair result = run_pair(
            model,
            prompt,
            options.new_tokens,
            (options.warmups + trial) % 2 == 0
        );
        cached_trials.push_back(result.cached_seconds);
        uncached_trials.push_back(result.uncached_seconds);
        progress << " cached=" << result.cached_seconds
                 << " s, uncached=" << result.uncached_seconds << " s\n";
    }

    return Results{
        model.config(),
        checkpoint_bytes,
        prompt,
        checkpoint_load_seconds,
        cached_trials,
        uncached_trials,
        median(cached_trials),
        median(uncached_trials),
    };
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
        const Results results = benchmark(options, std::cout);

        std::cout << '\n';
        print_results(std::cout, options, results.config, results);

        if (options.json_output.has_value()) {
            std::ofstream json(*options.json_output);
            if (!json) {
                throw std::runtime_error(
                    "could not open JSON output: " +
                    options.json_output->string()
                );
            }
            write_json(json, options, results.config, results);
            if (!json) {
                throw std::runtime_error(
                    "could not write JSON output: " +
                    options.json_output->string()
                );
            }
            std::cout << "JSON: " << options.json_output->string() << '\n';
        }
    } catch (const std::exception& exception) {
        std::cerr << "benchmark failed: " << exception.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
