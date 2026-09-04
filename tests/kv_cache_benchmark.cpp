// Measures what the key/value cache is worth, and confirms that both
// paths generate the same tokens while doing it.
//
// Usage:
//   gpt2_kv_cache_benchmark <checkpoint> [prompt tokens] [new tokens]

#include "gpt2/checkpoint.h"
#include "gpt2/generation.h"

#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

constexpr std::size_t default_prompt_tokens = 8;
constexpr std::size_t default_new_tokens = 24;

std::size_t parse_count(std::string_view text) {
    std::size_t value = 0;
    const char* const end = text.data() + text.size();
    const auto result = std::from_chars(text.data(), end, value);
    if (result.ec != std::errc{} || result.ptr != end) {
        throw std::invalid_argument(
            "invalid count: " + std::string(text)
        );
    }

    return value;
}

double seconds_since(
    std::chrono::steady_clock::time_point start
) {
    const std::chrono::duration<double> elapsed =
        std::chrono::steady_clock::now() - start;
    return elapsed.count();
}

}  // namespace

int main(int argument_count, char** arguments) {
    if (argument_count < 2 || argument_count > 4) {
        std::cerr << "usage: gpt2_kv_cache_benchmark <checkpoint> "
                     "[prompt tokens] [new tokens]\n";
        return EXIT_FAILURE;
    }

    try {
        const std::size_t prompt_tokens = argument_count > 2
            ? parse_count(arguments[2])
            : default_prompt_tokens;
        const std::size_t new_tokens = argument_count > 3
            ? parse_count(arguments[3])
            : default_new_tokens;

        const gpt2::Gpt2Model model(
            gpt2::load_checkpoint(arguments[1])
        );

        const std::size_t context_length =
            static_cast<std::size_t>(model.config().context_length);
        if (prompt_tokens == 0 || prompt_tokens >= context_length) {
            throw std::invalid_argument(
                "prompt tokens must be between one and context length "
                "minus one"
            );
        }
        if (new_tokens == 0) {
            throw std::invalid_argument(
                "new tokens must be greater than zero"
            );
        }

        // An arbitrary but fixed prompt: the benchmark measures speed,
        // and the tokens only have to be inside the vocabulary.
        std::vector<std::size_t> prompt(prompt_tokens);
        for (std::size_t index = 0; index < prompt_tokens; ++index) {
            prompt[index] = (index * 37 + 100) %
                static_cast<std::size_t>(model.config().vocab_size);
        }

        gpt2::GenerationLimits cached;
        cached.maximum_new_tokens = new_tokens;
        gpt2::GenerationLimits uncached = cached;
        uncached.use_cache = false;

        const auto cached_start = std::chrono::steady_clock::now();
        const gpt2::Generation with_cache =
            gpt2::generate_greedy(model, prompt, cached);
        const double cached_seconds = seconds_since(cached_start);

        const auto uncached_start = std::chrono::steady_clock::now();
        const gpt2::Generation without_cache =
            gpt2::generate_greedy(model, prompt, uncached);
        const double uncached_seconds = seconds_since(uncached_start);

        if (with_cache.new_token_ids != without_cache.new_token_ids) {
            throw std::runtime_error(
                "the cached and uncached paths generated different tokens"
            );
        }

        const double generated =
            static_cast<double>(with_cache.new_token_ids.size());

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "prompt tokens: " << prompt_tokens << '\n';
        std::cout << "new tokens: "
                  << with_cache.new_token_ids.size() << '\n';
        std::cout << "without cache: " << uncached_seconds << " s ("
                  << uncached_seconds / generated << " s/token)\n";
        std::cout << "with cache: " << cached_seconds << " s ("
                  << cached_seconds / generated << " s/token)\n";
        std::cout << "speedup: "
                  << uncached_seconds / cached_seconds << "x\n";
        std::cout << "tokens agree: yes\n";
    } catch (const std::exception& exception) {
        std::cerr << "gpt2_kv_cache_benchmark failed: "
                  << exception.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
