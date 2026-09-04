#include "cli.h"

#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct RunResult {
    int status;
    std::string output;
    std::string error;
};

int failure_count = 0;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failure_count;
    }
}

RunResult run(std::initializer_list<std::string_view> options) {
    std::vector<std::string> storage{"gpt2"};
    storage.reserve(options.size() + 1);
    for (const std::string_view option : options) {
        storage.emplace_back(option);
    }

    std::vector<char*> arguments;
    arguments.reserve(storage.size());
    for (std::string& argument : storage) {
        arguments.push_back(argument.data());
    }

    std::ostringstream output;
    std::ostringstream error;
    const int status = gpt2::run_cli(
        static_cast<int>(arguments.size()),
        arguments.data(),
        output,
        error
    );
    return RunResult{status, output.str(), error.str()};
}

void expect_error_contains(
    const RunResult& result,
    std::string_view expected,
    std::string_view message
) {
    expect(result.status == EXIT_FAILURE, message);
    expect(
        result.error.find(expected) != std::string::npos,
        message
    );
    expect(result.output.empty(), message);
}

void test_help() {
    const RunResult result = run({"--help"});
    expect(result.status == EXIT_SUCCESS, "--help succeeds");
    expect(
        result.output.find("Usage:") != std::string::npos,
        "--help prints the usage"
    );
    expect(result.error.empty(), "--help writes no error");

    const RunResult short_result = run({"-h"});
    expect(short_result.status == EXIT_SUCCESS, "-h succeeds");
}

void test_missing_and_unknown_options() {
    expect_error_contains(
        run({}),
        "missing required option --checkpoint",
        "the CLI requires a checkpoint"
    );
    expect_error_contains(
        run({"--unknown"}),
        "unknown option: --unknown",
        "the CLI rejects unknown options"
    );
    expect_error_contains(
        run({"--checkpoint"}),
        "--checkpoint requires a value",
        "the CLI rejects a missing option value"
    );
    expect_error_contains(
        run({"--prompt", "first", "--prompt", "second"}),
        "--prompt may only be provided once",
        "the CLI rejects duplicate options"
    );
}

void test_numeric_options() {
    expect_error_contains(
        run({"--max-new-tokens", "-1"}),
        "--max-new-tokens requires a non-negative integer",
        "the CLI rejects a negative token count"
    );
    expect_error_contains(
        run({"--top-k", "many"}),
        "--top-k requires a non-negative integer",
        "the CLI rejects a non-numeric top-k"
    );
    expect_error_contains(
        run({"--temperature", "hot"}),
        "--temperature requires a finite number",
        "the CLI rejects a non-numeric temperature"
    );
}

void test_sampling_options_require_sampling() {
    expect_error_contains(
        run({
            "--checkpoint", "missing.bin",
            "--vocab", "missing-vocab.json",
            "--merges", "missing-merges.txt",
            "--prompt", "Hello",
            "--temperature", "0.8",
        }),
        "sampling options require --sample",
        "sampling controls cannot be silently ignored"
    );
}

void test_sampling_ranges() {
    expect_error_contains(
        run({
            "--checkpoint", "missing.bin",
            "--vocab", "missing-vocab.json",
            "--merges", "missing-merges.txt",
            "--prompt", "Hello",
            "--sample",
            "--temperature", "0",
        }),
        "--temperature must be greater than zero",
        "the CLI rejects a zero temperature"
    );
    expect_error_contains(
        run({
            "--checkpoint", "missing.bin",
            "--vocab", "missing-vocab.json",
            "--merges", "missing-merges.txt",
            "--prompt", "Hello",
            "--sample",
            "--top-p", "1.1",
        }),
        "--top-p must be greater than zero and at most one",
        "the CLI rejects top-p above one"
    );
}

void test_prompt_and_file_errors() {
    expect_error_contains(
        run({
            "--checkpoint", "missing.bin",
            "--vocab", "missing-vocab.json",
            "--merges", "missing-merges.txt",
            "--prompt", "",
        }),
        "--prompt must not be empty",
        "the CLI rejects an empty prompt"
    );
    expect_error_contains(
        run({
            "--checkpoint", "missing.bin",
            "--vocab", "missing-vocab.json",
            "--merges", "missing-merges.txt",
            "--prompt", "Hello",
        }),
        "could not open tokenizer vocabulary",
        "valid options proceed to asset loading"
    );
}

}  // namespace

int main() {
    test_help();
    test_missing_and_unknown_options();
    test_numeric_options();
    test_sampling_options_require_sampling();
    test_sampling_ranges();
    test_prompt_and_file_errors();

    if (failure_count != 0) {
        std::cerr << failure_count << " CLI test assertion(s) failed\n";
        return EXIT_FAILURE;
    }

    std::cout << "All CLI tests passed\n";
    return EXIT_SUCCESS;
}
