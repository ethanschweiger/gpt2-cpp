#include "gpt2/tokenizer.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iomanip>
#include <ios>
#include <iostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

using TokenId = gpt2::Gpt2Tokenizer::TokenId;
using TokenIds = std::vector<TokenId>;

constexpr std::size_t byte_alphabet_size = 256;
constexpr std::size_t fixture_vocabulary_size = 268;

constexpr TokenId hello_id = 259;
constexpr TokenId or_id = 261;
constexpr TokenId ld_id = 263;
constexpr TokenId space_world_id = 264;
constexpr TokenId double_space_id = 266;
constexpr TokenId end_of_text_id = 267;

int failure_count = 0;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failure_count;
    }
}

std::string describe(const TokenIds& token_ids) {
    std::ostringstream stream;
    stream << '[';
    for (std::size_t index = 0; index < token_ids.size(); ++index) {
        if (index != 0) {
            stream << ", ";
        }
        stream << token_ids[index];
    }
    stream << ']';
    return stream.str();
}

std::string describe(std::string_view text) {
    std::ostringstream stream;
    stream << '"' << std::hex;
    for (const char character : text) {
        const auto byte = static_cast<unsigned char>(character);
        if (byte >= 0x20U && byte < 0x7FU) {
            stream << character;
        } else {
            stream << "\\x" << std::setw(2) << std::setfill('0')
                   << static_cast<unsigned int>(byte);
        }
    }
    stream << '"' << std::dec;
    return stream.str();
}

void expect_ids(
    const TokenIds& actual,
    const TokenIds& expected,
    std::string_view message
) {
    if (actual != expected) {
        std::cerr << "FAIL: " << message
                  << " (expected " << describe(expected)
                  << ", got " << describe(actual) << ")\n";
        ++failure_count;
    }
}

void expect_text(
    std::string_view actual,
    std::string_view expected,
    std::string_view message
) {
    if (actual != expected) {
        std::cerr << "FAIL: " << message
                  << " (expected " << describe(expected)
                  << ", got " << describe(actual) << ")\n";
        ++failure_count;
    }
}

template <typename ExpectedException, typename Function>
void expect_throws(Function function, std::string_view message) {
    try {
        function();
    } catch (const ExpectedException&) {
        return;
    } catch (const std::exception& exception) {
        std::cerr << "FAIL: " << message
                  << " (wrong exception: " << exception.what() << ")\n";
        ++failure_count;
        return;
    } catch (...) {
        std::cerr << "FAIL: " << message
                  << " (wrong non-standard exception)\n";
        ++failure_count;
        return;
    }

    std::cerr << "FAIL: " << message << " (no exception thrown)\n";
    ++failure_count;
}

std::string to_utf8(char32_t code_point) {
    std::string text;
    if (code_point <= 0x7FU) {
        text.push_back(static_cast<char>(code_point));
    } else if (code_point <= 0x7FFU) {
        text.push_back(static_cast<char>(0xC0U | (code_point >> 6U)));
        text.push_back(static_cast<char>(0x80U | (code_point & 0x3FU)));
    } else {
        text.push_back(static_cast<char>(0xE0U | (code_point >> 12U)));
        text.push_back(
            static_cast<char>(0x80U | ((code_point >> 6U) & 0x3FU))
        );
        text.push_back(static_cast<char>(0x80U | (code_point & 0x3FU)));
    }

    return text;
}

// An independent replica of GPT-2's bytes-to-unicode table, so the test
// does not depend on the implementation it exercises.
std::array<char32_t, byte_alphabet_size> make_byte_to_code_point() {
    std::array<char32_t, byte_alphabet_size> mapping{};
    std::array<bool, byte_alphabet_size> printable{};

    for (std::size_t byte = 0x21; byte <= 0x7E; ++byte) {
        printable[byte] = true;
    }
    for (std::size_t byte = 0xA1; byte <= 0xAC; ++byte) {
        printable[byte] = true;
    }
    for (std::size_t byte = 0xAE; byte <= 0xFF; ++byte) {
        printable[byte] = true;
    }

    char32_t next = 0x100;
    for (std::size_t byte = 0; byte < byte_alphabet_size; ++byte) {
        mapping[byte] = printable[byte]
            ? static_cast<char32_t>(byte)
            : next++;
    }

    return mapping;
}

const std::array<char32_t, byte_alphabet_size>& byte_to_code_point() {
    static const std::array<char32_t, byte_alphabet_size> mapping =
        make_byte_to_code_point();
    return mapping;
}

std::string alphabet_token(unsigned char byte) {
    return to_utf8(byte_to_code_point()[byte]);
}

const std::string& mapped_space() {
    static const std::string space =
        alphabet_token(static_cast<unsigned char>(' '));
    return space;
}

std::vector<std::string> fixture_extra_tokens() {
    const std::string& space = mapped_space();
    return {
        "he",
        "ll",
        "llo",
        "hello",
        space + "w",
        "or",
        space + "wor",
        "ld",
        space + "world",
        "n'",
        space + space,
        "<|endoftext|>",
    };
}

std::vector<std::string> fixture_merge_rules() {
    const std::string& space = mapped_space();
    return {
        "h e",
        "l l",
        "ll o",
        "he llo",
        space + " w",
        "o r",
        space + "w or",
        "l d",
        space + "wor ld",
        "n '",
        space + " " + space,
    };
}

std::string json_escape(std::string_view token) {
    std::string escaped;
    for (const char character : token) {
        if (character == '"' || character == '\\') {
            escaped.push_back('\\');
        }
        escaped.push_back(character);
    }

    return escaped;
}

// Emits the space character as a \u escape so that the loader's escape
// handling is exercised by the ordinary fixture.
std::string escaped_key(std::string_view token) {
    if (token == mapped_space()) {
        return "\\u0120";
    }

    return json_escape(token);
}

std::string make_vocabulary_json(
    const std::vector<std::string>& extra_tokens
) {
    std::ostringstream stream;
    stream << '{';

    for (std::size_t byte = 0; byte < byte_alphabet_size; ++byte) {
        if (byte != 0) {
            stream << ", ";
        }
        stream << '"'
               << escaped_key(alphabet_token(
                      static_cast<unsigned char>(byte)
                  ))
               << "\": " << byte;
    }

    for (std::size_t index = 0; index < extra_tokens.size(); ++index) {
        stream << ", \"" << json_escape(extra_tokens[index])
               << "\": " << (byte_alphabet_size + index);
    }

    stream << '}';
    return stream.str();
}

std::string make_merges_text(const std::vector<std::string>& rules) {
    std::string text = "#version: 0.2\n";
    for (const std::string& rule : rules) {
        text += rule;
        text.push_back('\n');
    }

    return text;
}

std::filesystem::path unique_temporary_path() {
    static std::size_t counter = 0;
    const auto timestamp = std::chrono::steady_clock::now()
        .time_since_epoch()
        .count();
    return std::filesystem::temp_directory_path() /
        ("gpt2-tokenizer-test-" + std::to_string(timestamp) +
         "-" + std::to_string(++counter));
}

class TemporaryAssets {
public:
    TemporaryAssets(
        std::string_view vocabulary,
        std::string_view merges
    )
        : directory_m(unique_temporary_path()) {
        std::filesystem::create_directories(directory_m);
        write(vocabulary_path(), vocabulary);
        write(merges_path(), merges);
    }

    TemporaryAssets(const TemporaryAssets&) = delete;
    TemporaryAssets& operator=(const TemporaryAssets&) = delete;

    ~TemporaryAssets() {
        std::error_code error;
        std::filesystem::remove_all(directory_m, error);
    }

    std::filesystem::path vocabulary_path() const {
        return directory_m / "vocab.json";
    }

    std::filesystem::path merges_path() const {
        return directory_m / "merges.txt";
    }

    const std::filesystem::path& directory() const {
        return directory_m;
    }

private:
    static void write(
        const std::filesystem::path& path,
        std::string_view contents
    ) {
        std::ofstream output(path, std::ios::binary);
        output.write(
            contents.data(),
            static_cast<std::streamsize>(contents.size())
        );
        if (!output) {
            throw std::runtime_error(
                "could not write tokenizer fixture: " + path.string()
            );
        }
    }

    std::filesystem::path directory_m;
};

gpt2::Gpt2Tokenizer load_fixture_tokenizer(const TemporaryAssets& assets) {
    return gpt2::Gpt2Tokenizer::load(
        assets.vocabulary_path(),
        assets.merges_path()
    );
}

TemporaryAssets make_fixture_assets() {
    return TemporaryAssets(
        make_vocabulary_json(fixture_extra_tokens()),
        make_merges_text(fixture_merge_rules())
    );
}

void expect_load_failure(
    std::string_view vocabulary,
    std::string_view merges,
    std::string_view message
) {
    expect_throws<std::runtime_error>(
        [&vocabulary, &merges] {
            const TemporaryAssets assets(vocabulary, merges);
            static_cast<void>(load_fixture_tokenizer(assets));
        },
        message
    );
}

void test_tokenizer_loads_and_reports_its_vocabulary() {
    const TemporaryAssets assets = make_fixture_assets();
    const gpt2::Gpt2Tokenizer tokenizer = load_fixture_tokenizer(assets);

    expect(
        tokenizer.vocabulary_size() == fixture_vocabulary_size,
        "tokenizer reports the size of its vocabulary"
    );

    expect_ids(
        tokenizer.encode("hello world"),
        {hello_id, space_world_id},
        "tokenizer loads a vocabulary written with \\u escapes"
    );
}

void test_byte_level_bpe_matches_the_reference_algorithm() {
    const TemporaryAssets assets = make_fixture_assets();
    const gpt2::Gpt2Tokenizer tokenizer = load_fixture_tokenizer(assets);

    struct Case {
        std::string_view text;
        TokenIds expected;
        std::string_view message;
    };

    const std::vector<Case> cases{
        {"", {}, "empty text produces no tokens"},
        {"a", {'a'}, "a single character maps to its alphabet token"},
        {
            "hello world",
            {hello_id, space_world_id},
            "greedy merges collapse a familiar phrase"
        },
        {
            " world",
            {space_world_id},
            "a leading space joins the following word"
        },
        {
            "world hello",
            {'w', or_id, ld_id, ' ', hello_id},
            "merges apply only where their ranks allow"
        },
        {
            "hello world hello",
            {hello_id, space_world_id, ' ', hello_id},
            "repeated phrases encode consistently"
        },
        {
            "hello\nworld",
            {hello_id, '\n', 'w', or_id, ld_id},
            "a newline separates its neighbours"
        },
        {
            "123 45",
            {'1', '2', '3', ' ', '4', '5'},
            "digits form their own pre-tokenizer pieces"
        },
    };

    for (const Case& test_case : cases) {
        expect_ids(
            tokenizer.encode(test_case.text),
            test_case.expected,
            test_case.message
        );
    }
}

void test_pre_tokenizer_splits_whitespace_like_the_reference() {
    const TemporaryAssets assets = make_fixture_assets();
    const gpt2::Gpt2Tokenizer tokenizer = load_fixture_tokenizer(assets);

    struct Case {
        std::string_view text;
        TokenIds expected;
        std::string_view message;
    };

    // The "\s+(?!\S)" alternative hands the final space of a run back to
    // " ?\p{L}+", so these cases pin down the backtracking behaviour.
    const std::vector<Case> cases{
        {
            "hello  world",
            {hello_id, ' ', space_world_id},
            "a doubled space splits into a lone space and a word"
        },
        {
            "  a",
            {' ', ' ', 'a'},
            "two spaces before a word do not merge"
        },
        {
            "   a",
            {double_space_id, ' ', 'a'},
            "three spaces before a word keep two together"
        },
        {
            "a  b",
            {'a', ' ', ' ', 'b'},
            "an interior doubled space splits around the words"
        },
        {
            "hello  ",
            {hello_id, double_space_id},
            "a trailing whitespace run stays whole"
        },
        {
            "abc   ",
            {'a', 'b', 'c', double_space_id, ' '},
            "a merge consumes non-overlapping pairs left to right"
        },
        {
            "  \n  ",
            {double_space_id, '\n', double_space_id},
            "a whitespace-only text forms a single piece"
        },
        {
            "\t\thello",
            {'\t', '\t', hello_id},
            "a tab run before a word splits into single tabs"
        },
        {
            "ab\vcd",
            {'a', 'b', '\v', 'c', 'd'},
            "a lone vertical tab falls through to the final alternative"
        },
    };

    for (const Case& test_case : cases) {
        expect_ids(
            tokenizer.encode(test_case.text),
            test_case.expected,
            test_case.message
        );
    }
}

void test_pre_tokenizer_handles_contractions() {
    const TemporaryAssets assets = make_fixture_assets();
    const gpt2::Gpt2Tokenizer tokenizer = load_fixture_tokenizer(assets);

    // "n '" is a merge rule, so it fires only if the apostrophe wrongly
    // shares a piece with the preceding word.
    expect_ids(
        tokenizer.encode("don't"),
        {'d', 'o', 'n', '\'', 't'},
        "a lowercase contraction becomes its own piece"
    );
    expect_ids(
        tokenizer.encode("Don'T"),
        {'D', 'o', 'n', '\'', 'T'},
        "contractions are matched case-sensitively"
    );
    // "'ll" is one piece, so the "l l" merge applies inside it.
    expect_ids(
        tokenizer.encode("'ll"),
        {'\'', 257},
        "a leading contraction is a piece of its own"
    );
    expect_ids(
        tokenizer.encode("'s"),
        {'\'', 's'},
        "a two-character contraction matches"
    );
    expect_ids(
        tokenizer.encode("'x"),
        {'\'', 'x'},
        "an apostrophe without a contraction is punctuation"
    );
    expect_ids(
        tokenizer.encode("'"),
        {'\''},
        "a trailing apostrophe cannot start a contraction"
    );
}

void test_multi_byte_text_round_trips() {
    const TemporaryAssets assets = make_fixture_assets();
    const gpt2::Gpt2Tokenizer tokenizer = load_fixture_tokenizer(assets);

    expect_ids(
        tokenizer.encode("\u00e9"),
        {0xC3, 0xA9},
        "a two-byte character encodes as its mapped bytes"
    );
    expect_ids(
        tokenizer.encode("caf\u00e9"),
        {'c', 'a', 'f', 0xC3, 0xA9},
        "a Latin-1 accent stays inside its word piece"
    );
    expect_ids(
        tokenizer.encode("\U0001f642"),
        {0xF0, 0x9F, 0x99, 0x82},
        "a four-byte character encodes as its mapped bytes"
    );

    for (const std::string_view text : {
             "hello world",
             "caf\u00e9",
             "\U0001f642",
             "\u65e5\u672c\u8a9e",
             "   a",
             "  \n  ",
             "don't",
         }) {
        const TokenIds token_ids = tokenizer.encode(text);
        expect_text(
            tokenizer.decode(token_ids),
            text,
            "encode and decode round trip"
        );
    }
}

void test_end_of_text_marker_is_a_single_token() {
    const TemporaryAssets assets = make_fixture_assets();
    const gpt2::Gpt2Tokenizer tokenizer = load_fixture_tokenizer(assets);

    expect_ids(
        tokenizer.encode("<|endoftext|>"),
        {end_of_text_id},
        "the marker encodes as one token"
    );
    expect_ids(
        tokenizer.encode("a<|endoftext|>b"),
        {'a', end_of_text_id, 'b'},
        "the marker splits the surrounding text"
    );
    expect_ids(
        tokenizer.encode("<|endoftext|><|endoftext|>"),
        {end_of_text_id, end_of_text_id},
        "adjacent markers each encode separately"
    );
    expect_ids(
        tokenizer.encode("hello world<|endoftext|> world"),
        {hello_id, space_world_id, end_of_text_id, space_world_id},
        "text around the marker still merges normally"
    );
    expect_ids(
        tokenizer.encode("<|endoftext|"),
        {
            '<', '|', 'e', 'n', 'd', 'o', 'f',
            't', 'e', 'x', 't', '|'
        },
        "an incomplete marker is ordinary text"
    );
    expect_text(
        tokenizer.decode(std::array<TokenId, 1>{end_of_text_id}),
        "<|endoftext|>",
        "the marker decodes back to its literal text"
    );
}

void test_decode_reports_partial_and_invalid_sequences() {
    const TemporaryAssets assets = make_fixture_assets();
    const gpt2::Gpt2Tokenizer tokenizer = load_fixture_tokenizer(assets);

    const std::span<const TokenId> empty;
    expect_text(
        tokenizer.decode(empty),
        "",
        "an empty sequence decodes to empty text"
    );

    // A truncated multi-byte character decodes to the bytes its tokens
    // carry, which is not by itself valid UTF-8.
    const std::array<TokenId, 1> partial{0xC3};
    expect_text(
        tokenizer.decode(partial),
        "\xC3",
        "a truncated character decodes to its leading byte"
    );

    const std::array<TokenId, 1> unknown{fixture_vocabulary_size};
    expect_throws<std::out_of_range>(
        [&tokenizer, &unknown] {
            static_cast<void>(tokenizer.decode(unknown));
        },
        "decode rejects a token ID outside the vocabulary"
    );
}

void test_encode_rejects_invalid_utf8() {
    const TemporaryAssets assets = make_fixture_assets();
    const gpt2::Gpt2Tokenizer tokenizer = load_fixture_tokenizer(assets);

    for (const std::string_view text : {
             "\xFF",
             "\xC3",
             "\xC0\xAF",
             "\xE0\x80\xAF",
             "\xED\xA0\x80",
             "\xF5\x80\x80\x80",
             "a\x80""b",
         }) {
        expect_throws<std::invalid_argument>(
            [&tokenizer, text] {
                static_cast<void>(tokenizer.encode(text));
            },
            "encode rejects invalid UTF-8"
        );
    }
}

void test_loader_rejects_invalid_assets() {
    const std::string valid_vocabulary =
        make_vocabulary_json(fixture_extra_tokens());
    const std::string valid_merges =
        make_merges_text(fixture_merge_rules());

    expect_throws<std::runtime_error>(
        [] {
            static_cast<void>(gpt2::Gpt2Tokenizer::load(
                unique_temporary_path() / "vocab.json",
                unique_temporary_path() / "merges.txt"
            ));
        },
        "loader reports a missing vocabulary file"
    );

    expect_load_failure(
        "{\"a\": 0,}",
        valid_merges,
        "loader rejects a trailing comma in the vocabulary"
    );
    expect_load_failure(
        "not json",
        valid_merges,
        "loader rejects a vocabulary that is not an object"
    );
    expect_load_failure(
        valid_vocabulary + "{}",
        valid_merges,
        "loader rejects trailing content after the vocabulary"
    );
    expect_load_failure(
        "{\"a\": 0, \"b\": 0}",
        valid_merges,
        "loader rejects a vocabulary that reuses an ID"
    );
    expect_load_failure(
        "{\"a\": 0, \"b\": 2}",
        valid_merges,
        "loader rejects vocabulary IDs with a gap"
    );
    expect_load_failure(
        "{\"\\ud800\": 0}",
        valid_merges,
        "loader rejects an unpaired surrogate escape"
    );
    expect_load_failure(
        "{\"\\ud83d\\ude42\": 0}",
        valid_merges,
        "loader rejects a token outside the byte-level alphabet"
    );

    {
        std::vector<std::string> missing_marker = fixture_extra_tokens();
        missing_marker.pop_back();
        expect_load_failure(
            make_vocabulary_json(missing_marker),
            valid_merges,
            "loader rejects a vocabulary without the end-of-text marker"
        );
    }

    expect_load_failure(
        valid_vocabulary,
        "#version: 0.2\n",
        "loader rejects a merges file with no rules"
    );
    expect_load_failure(
        valid_vocabulary,
        "#version: 0.2\nh e l\n",
        "loader rejects a merges line with three symbols"
    );
    expect_load_failure(
        valid_vocabulary,
        "#version: 0.2\nhe\n",
        "loader rejects a merges line with one symbol"
    );
    expect_load_failure(
        valid_vocabulary,
        "#version: 0.2\nh e\nh e\n",
        "loader rejects a repeated merge rule"
    );
    expect_load_failure(
        valid_vocabulary,
        "#version: 0.2\nh o\n",
        "loader rejects a merge outside the vocabulary"
    );
}

void test_loader_accepts_asset_layout_variations() {
    const std::vector<std::string> rules = fixture_merge_rules();
    const std::string valid_vocabulary =
        make_vocabulary_json(fixture_extra_tokens());

    {
        // A merges file without the "#version" header keeps every line.
        std::string merges;
        for (const std::string& rule : rules) {
            merges += rule;
            merges.push_back('\n');
        }

        const TemporaryAssets assets(valid_vocabulary, merges);
        const gpt2::Gpt2Tokenizer tokenizer =
            load_fixture_tokenizer(assets);
        expect_ids(
            tokenizer.encode("hello world"),
            {hello_id, space_world_id},
            "loader keeps the first rule when no version header exists"
        );
    }

    {
        // Windows line endings and a missing final newline.
        std::string merges = "#version: 0.2\r\n";
        for (std::size_t index = 0; index < rules.size(); ++index) {
            merges += rules[index];
            if (index + 1 != rules.size()) {
                merges += "\r\n";
            }
        }

        const TemporaryAssets assets(valid_vocabulary, merges);
        const gpt2::Gpt2Tokenizer tokenizer =
            load_fixture_tokenizer(assets);
        expect_ids(
            tokenizer.encode("hello world"),
            {hello_id, space_world_id},
            "loader accepts carriage returns and no trailing newline"
        );
    }

    {
        // Whitespace between JSON tokens is insignificant.
        std::string spaced = "  {\n";
        for (std::size_t byte = 0; byte < byte_alphabet_size; ++byte) {
            if (byte != 0) {
                spaced += ",\n";
            }
            spaced += "    \"" +
                escaped_key(alphabet_token(
                    static_cast<unsigned char>(byte)
                )) +
                "\" : " + std::to_string(byte);
        }

        const std::vector<std::string> extra = fixture_extra_tokens();
        for (std::size_t index = 0; index < extra.size(); ++index) {
            spaced += ",\n    \"" + json_escape(extra[index]) +
                "\" : " + std::to_string(byte_alphabet_size + index);
        }
        spaced += "\n  }\n";

        const TemporaryAssets assets(
            spaced,
            make_merges_text(rules)
        );
        const gpt2::Gpt2Tokenizer tokenizer =
            load_fixture_tokenizer(assets);
        expect(
            tokenizer.vocabulary_size() == fixture_vocabulary_size,
            "loader accepts a pretty-printed vocabulary"
        );
    }
}

}  // namespace

int main() {
    try {
        test_tokenizer_loads_and_reports_its_vocabulary();
        test_byte_level_bpe_matches_the_reference_algorithm();
        test_pre_tokenizer_splits_whitespace_like_the_reference();
        test_pre_tokenizer_handles_contractions();
        test_multi_byte_text_round_trips();
        test_end_of_text_marker_is_a_single_token();
        test_decode_reports_partial_and_invalid_sequences();
        test_encode_rejects_invalid_utf8();
        test_loader_rejects_invalid_assets();
        test_loader_accepts_asset_layout_variations();
    } catch (const std::exception& exception) {
        std::cerr << "FAIL: unexpected exception: "
                  << exception.what() << '\n';
        return EXIT_FAILURE;
    }

    if (failure_count != 0) {
        std::cerr << failure_count
                  << " tokenizer test assertion(s) failed\n";
        return EXIT_FAILURE;
    }

    std::cout << "All tokenizer tests passed\n";
    return EXIT_SUCCESS;
}
