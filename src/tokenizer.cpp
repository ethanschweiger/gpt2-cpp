#include "gpt2/tokenizer.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <ios>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gpt2 {

namespace {

struct CodePointRange {
    char32_t first;
    char32_t last;
};

#include "unicode_ranges.inc"

using Vocabulary =
    std::unordered_map<std::string, Gpt2Tokenizer::TokenId>;
using MergeRanks = std::unordered_map<std::string, std::size_t>;

constexpr std::size_t byte_alphabet_size = 256;
constexpr char32_t remapped_byte_count = 68;
constexpr char32_t first_remapped_code_point = 0x100;
constexpr char32_t code_point_limit =
    first_remapped_code_point + remapped_byte_count;
constexpr char32_t maximum_code_point = 0x10FFFF;
constexpr char32_t first_high_surrogate = 0xD800;
constexpr char32_t last_high_surrogate = 0xDBFF;
constexpr char32_t first_low_surrogate = 0xDC00;
constexpr char32_t last_low_surrogate = 0xDFFF;
constexpr char32_t first_supplementary_code_point = 0x10000;

constexpr std::string_view end_of_text_token = "<|endoftext|>";
constexpr std::string_view merges_version_prefix = "#version:";
constexpr std::string_view vocabulary_description = "tokenizer vocabulary";
constexpr std::string_view merges_description = "tokenizer merges";

constexpr std::size_t maximum_vocabulary_size = std::size_t{1} << 22U;
constexpr std::size_t maximum_token_length = std::size_t{1} << 16U;
constexpr std::uint64_t maximum_asset_size =
    std::uint64_t{256} * 1024U * 1024U;

// The leading alternatives of the GPT-2 pre-tokenizer pattern, in the
// order a backtracking engine tries them.
constexpr std::array<std::string_view, 7> contractions{
    "'s", "'t", "'re", "'ve", "'m", "'ll", "'d"
};

std::string read_text_file(
    const std::filesystem::path& path,
    std::string_view description
) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error(
            "could not open " + std::string(description) + ": " +
            path.string()
        );
    }

    const std::ifstream::pos_type end_position = input.tellg();
    if (end_position == std::ifstream::pos_type{-1}) {
        throw std::runtime_error(
            "could not determine the size of " +
            std::string(description) + ": " + path.string()
        );
    }

    const std::streamoff signed_size =
        static_cast<std::streamoff>(end_position);
    if (signed_size < 0) {
        throw std::runtime_error(
            "could not determine the size of " +
            std::string(description) + ": " + path.string()
        );
    }

    const std::uint64_t file_size =
        static_cast<std::uint64_t>(signed_size);
    if (file_size > maximum_asset_size) {
        throw std::runtime_error(
            std::string(description) + " exceeds the loader limit: " +
            path.string()
        );
    }

    input.seekg(0, std::ios::beg);
    if (!input) {
        throw std::runtime_error(
            "could not rewind " + std::string(description) + ": " +
            path.string()
        );
    }

    std::string contents(static_cast<std::size_t>(file_size), '\0');
    if (file_size != 0) {
        input.read(
            contents.data(),
            static_cast<std::streamsize>(file_size)
        );
        if (!input) {
            throw std::runtime_error(
                "could not read " + std::string(description) + ": " +
                path.string()
            );
        }
    }

    return contents;
}

template <std::size_t Count>
bool contains_code_point(
    const std::array<CodePointRange, Count>& ranges,
    char32_t value
) {
    const auto position = std::upper_bound(
        ranges.begin(),
        ranges.end(),
        value,
        [](char32_t probe, const CodePointRange& range) {
            return probe < range.first;
        }
    );
    if (position == ranges.begin()) {
        return false;
    }

    return value <= (position - 1)->last;
}

bool is_letter(char32_t value) {
    return contains_code_point(letter_ranges, value);
}

bool is_number(char32_t value) {
    return contains_code_point(number_ranges, value);
}

bool is_whitespace(char32_t value) {
    return contains_code_point(whitespace_ranges, value);
}

bool is_other(char32_t value) {
    return !is_whitespace(value) &&
        !is_letter(value) &&
        !is_number(value);
}

// GPT-2 maps every byte onto a printable code point so that byte-level
// BPE can be expressed over ordinary text. Printable Latin-1 bytes map
// to themselves and the remaining 68 bytes map to U+0100 upward, in
// increasing byte order.
struct ByteAlphabet {
    std::array<char32_t, byte_alphabet_size> byte_to_code_point{};
    std::array<std::int16_t, code_point_limit> code_point_to_byte{};
};

ByteAlphabet make_byte_alphabet() {
    ByteAlphabet alphabet;
    alphabet.code_point_to_byte.fill(-1);

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

    char32_t next_remapped = first_remapped_code_point;
    for (std::size_t byte = 0; byte < byte_alphabet_size; ++byte) {
        const char32_t code_point = printable[byte]
            ? static_cast<char32_t>(byte)
            : next_remapped++;
        alphabet.byte_to_code_point[byte] = code_point;
        alphabet.code_point_to_byte[code_point] =
            static_cast<std::int16_t>(byte);
    }

    if (next_remapped != code_point_limit) {
        throw std::logic_error(
            "byte-level alphabet has an unexpected size"
        );
    }

    return alphabet;
}

const ByteAlphabet& byte_alphabet() {
    static const ByteAlphabet alphabet = make_byte_alphabet();
    return alphabet;
}

void append_utf8(std::string& text, char32_t code_point) {
    const auto emit = [&text](char32_t value) {
        text.push_back(static_cast<char>(
            static_cast<unsigned char>(value)
        ));
    };

    if (code_point <= 0x7FU) {
        emit(code_point);
    } else if (code_point <= 0x7FFU) {
        emit(0xC0U | (code_point >> 6U));
        emit(0x80U | (code_point & 0x3FU));
    } else if (code_point <= 0xFFFFU) {
        emit(0xE0U | (code_point >> 12U));
        emit(0x80U | ((code_point >> 6U) & 0x3FU));
        emit(0x80U | (code_point & 0x3FU));
    } else {
        emit(0xF0U | (code_point >> 18U));
        emit(0x80U | ((code_point >> 12U) & 0x3FU));
        emit(0x80U | ((code_point >> 6U) & 0x3FU));
        emit(0x80U | (code_point & 0x3FU));
    }
}

struct DecodedText {
    std::vector<char32_t> code_points;
    // One byte offset per code point, plus the end of the input.
    std::vector<std::size_t> byte_offsets;
};

[[noreturn]] void throw_invalid_utf8(std::string_view description) {
    throw std::invalid_argument(
        std::string(description) + " is not valid UTF-8"
    );
}

DecodedText decode_utf8(
    std::string_view text,
    std::string_view description
) {
    DecodedText decoded;
    decoded.code_points.reserve(text.size());
    decoded.byte_offsets.reserve(text.size() + 1);

    std::size_t index = 0;
    while (index < text.size()) {
        const auto first = static_cast<unsigned char>(text[index]);
        std::size_t continuation_count = 0;
        char32_t code_point = 0;
        unsigned char second_minimum = 0x80U;
        unsigned char second_maximum = 0xBFU;

        if (first <= 0x7FU) {
            code_point = static_cast<char32_t>(first);
        } else if (first >= 0xC2U && first <= 0xDFU) {
            continuation_count = 1;
            code_point = static_cast<char32_t>(first & 0x1FU);
        } else if (first >= 0xE0U && first <= 0xEFU) {
            continuation_count = 2;
            code_point = static_cast<char32_t>(first & 0x0FU);
            if (first == 0xE0U) {
                second_minimum = 0xA0U;
            } else if (first == 0xEDU) {
                second_maximum = 0x9FU;
            }
        } else if (first >= 0xF0U && first <= 0xF4U) {
            continuation_count = 3;
            code_point = static_cast<char32_t>(first & 0x07U);
            if (first == 0xF0U) {
                second_minimum = 0x90U;
            } else if (first == 0xF4U) {
                second_maximum = 0x8FU;
            }
        } else {
            throw_invalid_utf8(description);
        }

        if (continuation_count > text.size() - index - 1) {
            throw_invalid_utf8(description);
        }

        for (std::size_t offset = 1;
             offset <= continuation_count;
             ++offset) {
            const auto byte =
                static_cast<unsigned char>(text[index + offset]);
            const unsigned char minimum =
                offset == 1 ? second_minimum : 0x80U;
            const unsigned char maximum =
                offset == 1 ? second_maximum : 0xBFU;
            if (byte < minimum || byte > maximum) {
                throw_invalid_utf8(description);
            }

            code_point = static_cast<char32_t>(
                (code_point << 6U) | (byte & 0x3FU)
            );
        }

        if (code_point > maximum_code_point) {
            throw_invalid_utf8(description);
        }

        decoded.byte_offsets.push_back(index);
        decoded.code_points.push_back(code_point);
        index += continuation_count + 1;
    }

    decoded.byte_offsets.push_back(text.size());
    return decoded;
}

using CodePointPredicate = bool (*)(char32_t);

// Implements " ?<class>+": the optional leading space is taken only
// when at least one class member follows it.
std::size_t match_run(
    const std::vector<char32_t>& code_points,
    std::size_t start,
    CodePointPredicate belongs
) {
    std::size_t index = start;
    if (code_points[index] == U' ' &&
        index + 1 < code_points.size() &&
        belongs(code_points[index + 1])) {
        ++index;
    }

    if (!belongs(code_points[index])) {
        return start;
    }

    while (index < code_points.size() && belongs(code_points[index])) {
        ++index;
    }

    return index;
}

// Returns the end of the piece the GPT-2 pre-tokenizer matches at
// `start`. The pattern is
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+
//   |\s+(?!\S)|\s+
// and its alternatives are tried in order, exactly as a backtracking
// engine would try them.
std::size_t match_piece(
    const std::vector<char32_t>& code_points,
    std::size_t start
) {
    const std::size_t size = code_points.size();

    if (code_points[start] == U'\'') {
        for (const std::string_view contraction : contractions) {
            if (size - start < contraction.size()) {
                continue;
            }

            bool matches = true;
            for (std::size_t offset = 1;
                 offset < contraction.size();
                 ++offset) {
                const auto expected = static_cast<char32_t>(
                    static_cast<unsigned char>(contraction[offset])
                );
                if (code_points[start + offset] != expected) {
                    matches = false;
                    break;
                }
            }

            if (matches) {
                return start + contraction.size();
            }
        }
    }

    for (const CodePointPredicate belongs :
         {is_letter, is_number, is_other}) {
        const std::size_t end = match_run(code_points, start, belongs);
        if (end != start) {
            return end;
        }
    }

    if (is_whitespace(code_points[start])) {
        std::size_t run_end = start;
        while (run_end < size && is_whitespace(code_points[run_end])) {
            ++run_end;
        }

        // "\s+(?!\S)" keeps a run that reaches the end of the input and
        // otherwise gives its last character back so that " ?\p{L}+"
        // can claim it. A lone whitespace character before a
        // non-whitespace one falls through to the trailing "\s+".
        if (run_end == size) {
            return run_end;
        }
        if (run_end - start >= 2) {
            return run_end - 1;
        }

        return run_end;
    }

    throw std::logic_error("GPT-2 pre-tokenizer failed to advance");
}

class JsonReader {
public:
    JsonReader(std::string_view text, std::string_view description)
        : text_m(text), description_m(description) {}

    void skip_whitespace() {
        while (position_m < text_m.size()) {
            const char character = text_m[position_m];
            if (character != ' ' && character != '\t' &&
                character != '\n' && character != '\r') {
                break;
            }
            ++position_m;
        }
    }

    bool try_consume(char expected) {
        skip_whitespace();
        if (position_m < text_m.size() &&
            text_m[position_m] == expected) {
            ++position_m;
            return true;
        }

        return false;
    }

    void expect(char expected) {
        if (!try_consume(expected)) {
            fail(std::string("expected '") + expected + "'");
        }
    }

    bool at_end() {
        skip_whitespace();
        return position_m >= text_m.size();
    }

    std::string read_string() {
        expect('"');
        std::string value;

        while (true) {
            if (position_m >= text_m.size()) {
                fail("a string is not terminated");
            }

            const auto byte =
                static_cast<unsigned char>(text_m[position_m]);
            ++position_m;

            if (byte == static_cast<unsigned char>('"')) {
                return value;
            }
            if (byte < 0x20U) {
                fail("a string contains an unescaped control character");
            }
            if (byte != static_cast<unsigned char>('\\')) {
                value.push_back(static_cast<char>(byte));
                continue;
            }

            read_escape(value);
        }
    }

    std::uint64_t read_unsigned() {
        skip_whitespace();
        const std::size_t start = position_m;
        while (position_m < text_m.size() &&
               text_m[position_m] >= '0' &&
               text_m[position_m] <= '9') {
            ++position_m;
        }

        const std::size_t digit_count = position_m - start;
        if (digit_count == 0) {
            fail("expected a non-negative integer");
        }
        if (digit_count > 1 && text_m[start] == '0') {
            fail("an integer has a redundant leading zero");
        }

        std::uint64_t value = 0;
        for (std::size_t index = start; index < position_m; ++index) {
            const auto digit = static_cast<std::uint64_t>(
                text_m[index] - '0'
            );
            if (value >
                (std::numeric_limits<std::uint64_t>::max() - digit) /
                    10U) {
                fail("an integer is out of range");
            }
            value = value * 10U + digit;
        }

        return value;
    }

    [[noreturn]] void fail(std::string_view reason) const {
        throw std::runtime_error(
            std::string(description_m) + " is malformed at byte " +
            std::to_string(position_m) + ": " + std::string(reason)
        );
    }

private:
    void read_escape(std::string& value) {
        if (position_m >= text_m.size()) {
            fail("a string ends with an incomplete escape");
        }

        const char escape = text_m[position_m];
        ++position_m;

        switch (escape) {
        case '"':
        case '\\':
        case '/':
            value.push_back(escape);
            return;
        case 'b':
            value.push_back('\b');
            return;
        case 'f':
            value.push_back('\f');
            return;
        case 'n':
            value.push_back('\n');
            return;
        case 'r':
            value.push_back('\r');
            return;
        case 't':
            value.push_back('\t');
            return;
        case 'u':
            break;
        default:
            fail("a string contains an unknown escape");
        }

        char32_t code_point = read_hex_quad();
        if (code_point >= first_high_surrogate &&
            code_point <= last_high_surrogate) {
            if (position_m + 1 >= text_m.size() ||
                text_m[position_m] != '\\' ||
                text_m[position_m + 1] != 'u') {
                fail("a string has an unpaired high surrogate");
            }
            position_m += 2;

            const char32_t low = read_hex_quad();
            if (low < first_low_surrogate || low > last_low_surrogate) {
                fail("a string has an unpaired high surrogate");
            }

            code_point = first_supplementary_code_point +
                ((code_point - first_high_surrogate) << 10U) +
                (low - first_low_surrogate);
        } else if (code_point >= first_low_surrogate &&
                   code_point <= last_low_surrogate) {
            fail("a string has an unpaired low surrogate");
        }

        append_utf8(value, code_point);
    }

    char32_t read_hex_quad() {
        if (text_m.size() - position_m < std::size_t{4}) {
            fail("a string has a truncated \\u escape");
        }

        char32_t value = 0;
        for (std::size_t offset = 0; offset < 4; ++offset) {
            const char digit = text_m[position_m + offset];
            char32_t nibble = 0;
            if (digit >= '0' && digit <= '9') {
                nibble = static_cast<char32_t>(digit - '0');
            } else if (digit >= 'a' && digit <= 'f') {
                nibble = static_cast<char32_t>(digit - 'a' + 10);
            } else if (digit >= 'A' && digit <= 'F') {
                nibble = static_cast<char32_t>(digit - 'A' + 10);
            } else {
                fail("a string has a malformed \\u escape");
            }
            value = static_cast<char32_t>((value << 4U) | nibble);
        }

        position_m += 4;
        return value;
    }

    std::string_view text_m;
    std::string_view description_m;
    std::size_t position_m = 0;
};

void require_alphabet_token(
    std::string_view token,
    std::string_view description
) {
    if (token.empty()) {
        throw std::runtime_error(
            std::string(description) + " contains an empty token"
        );
    }
    if (token.size() > maximum_token_length) {
        throw std::runtime_error(
            std::string(description) +
            " contains a token that exceeds the loader limit"
        );
    }

    DecodedText decoded;
    try {
        decoded = decode_utf8(token, description);
    } catch (const std::invalid_argument& exception) {
        throw std::runtime_error(exception.what());
    }

    const ByteAlphabet& alphabet = byte_alphabet();
    for (const char32_t code_point : decoded.code_points) {
        if (code_point >= code_point_limit ||
            alphabet.code_point_to_byte[code_point] < 0) {
            throw std::runtime_error(
                std::string(description) +
                " contains a token outside the byte-level alphabet"
            );
        }
    }
}

Vocabulary parse_vocabulary(std::string_view contents) {
    JsonReader reader(contents, vocabulary_description);
    Vocabulary token_to_id;
    reader.expect('{');

    if (!reader.try_consume('}')) {
        while (true) {
            std::string token = reader.read_string();
            require_alphabet_token(token, vocabulary_description);
            reader.expect(':');

            const std::uint64_t id = reader.read_unsigned();
            if (id >= maximum_vocabulary_size) {
                reader.fail("a token ID exceeds the loader limit");
            }

            const bool inserted = token_to_id.emplace(
                std::move(token),
                static_cast<Gpt2Tokenizer::TokenId>(id)
            ).second;
            if (!inserted) {
                reader.fail("the vocabulary contains a duplicate token");
            }
            if (token_to_id.size() > maximum_vocabulary_size) {
                reader.fail("the vocabulary exceeds the loader limit");
            }

            if (!reader.try_consume(',')) {
                break;
            }
        }

        reader.expect('}');
    }

    if (!reader.at_end()) {
        reader.fail("the vocabulary has trailing content");
    }

    if (token_to_id.empty()) {
        throw std::runtime_error("tokenizer vocabulary is empty");
    }

    return token_to_id;
}

std::vector<std::string> build_id_to_token(
    const Vocabulary& token_to_id
) {
    std::vector<std::string> id_to_token(token_to_id.size());
    std::vector<bool> assigned(token_to_id.size(), false);

    for (const auto& [token, id] : token_to_id) {
        if (id >= id_to_token.size()) {
            throw std::runtime_error(
                "tokenizer vocabulary IDs are not a dense range that "
                "starts at zero"
            );
        }
        if (assigned[id]) {
            throw std::runtime_error(
                "tokenizer vocabulary assigns one ID to several tokens"
            );
        }

        assigned[id] = true;
        id_to_token[id] = token;
    }

    return id_to_token;
}

void require_complete_alphabet(const Vocabulary& token_to_id) {
    const ByteAlphabet& alphabet = byte_alphabet();
    std::string token;

    for (std::size_t byte = 0; byte < byte_alphabet_size; ++byte) {
        token.clear();
        append_utf8(token, alphabet.byte_to_code_point[byte]);
        if (token_to_id.find(token) == token_to_id.end()) {
            throw std::runtime_error(
                "tokenizer vocabulary is missing the byte-level "
                "alphabet character for byte " + std::to_string(byte)
            );
        }
    }
}

MergeRanks parse_merges(
    std::string_view contents,
    const Vocabulary& token_to_id
) {
    MergeRanks merge_ranks;
    std::size_t line_start = 0;
    std::size_t line_number = 0;

    while (true) {
        std::size_t line_end = contents.find('\n', line_start);
        const bool is_final_line = line_end == std::string_view::npos;
        if (is_final_line) {
            line_end = contents.size();
        }

        std::string_view line = contents.substr(
            line_start,
            line_end - line_start
        );
        if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1);
        }

        ++line_number;
        const bool is_version_header = line_number == 1 &&
            line.starts_with(merges_version_prefix);

        if (!line.empty() && !is_version_header) {
            const std::size_t separator = line.find(' ');
            if (separator == std::string_view::npos ||
                line.find(' ', separator + 1) !=
                    std::string_view::npos) {
                throw std::runtime_error(
                    "tokenizer merges line " +
                    std::to_string(line_number) +
                    " is not a pair of symbols"
                );
            }

            const std::string_view left = line.substr(0, separator);
            const std::string_view right = line.substr(separator + 1);
            require_alphabet_token(left, merges_description);
            require_alphabet_token(right, merges_description);

            std::string merged;
            merged.reserve(left.size() + right.size());
            merged.append(left);
            merged.append(right);
            if (token_to_id.find(merged) == token_to_id.end()) {
                throw std::runtime_error(
                    "tokenizer merges line " +
                    std::to_string(line_number) +
                    " produces a token outside the vocabulary"
                );
            }

            const bool inserted = merge_ranks.emplace(
                std::string(line),
                merge_ranks.size()
            ).second;
            if (!inserted) {
                throw std::runtime_error(
                    "tokenizer merges line " +
                    std::to_string(line_number) +
                    " repeats an earlier rule"
                );
            }
        }

        if (is_final_line) {
            break;
        }
        line_start = line_end + 1;
    }

    if (merge_ranks.empty()) {
        throw std::runtime_error(
            "tokenizer merges file has no merge rules"
        );
    }

    return merge_ranks;
}

}  // namespace

Gpt2Tokenizer::Gpt2Tokenizer(
    Vocabulary token_to_id,
    std::vector<std::string> id_to_token,
    MergeRanks merge_ranks,
    TokenId end_of_text_id
)
    : token_to_id_m(std::move(token_to_id)),
      id_to_token_m(std::move(id_to_token)),
      merge_ranks_m(std::move(merge_ranks)),
      end_of_text_id_m(end_of_text_id) {}

Gpt2Tokenizer Gpt2Tokenizer::load(
    const std::filesystem::path& vocabulary_path,
    const std::filesystem::path& merges_path
) {
    Vocabulary token_to_id = parse_vocabulary(
        read_text_file(vocabulary_path, vocabulary_description)
    );
    std::vector<std::string> id_to_token = build_id_to_token(token_to_id);
    require_complete_alphabet(token_to_id);

    MergeRanks merge_ranks = parse_merges(
        read_text_file(merges_path, merges_description),
        token_to_id
    );

    const auto end_of_text =
        token_to_id.find(std::string(end_of_text_token));
    if (end_of_text == token_to_id.end()) {
        throw std::runtime_error(
            "tokenizer vocabulary does not contain " +
            std::string(end_of_text_token)
        );
    }
    const TokenId end_of_text_id = end_of_text->second;

    return Gpt2Tokenizer(
        std::move(token_to_id),
        std::move(id_to_token),
        std::move(merge_ranks),
        end_of_text_id
    );
}

std::size_t Gpt2Tokenizer::vocabulary_size() const {
    return id_to_token_m.size();
}

std::vector<std::string> Gpt2Tokenizer::apply_bpe(
    std::string_view piece
) const {
    // Every byte-level alphabet character encodes as one or two UTF-8
    // bytes, so the initial symbols are the characters of `piece`.
    std::vector<std::string> symbols;
    for (std::size_t index = 0; index < piece.size();) {
        const auto lead = static_cast<unsigned char>(piece[index]);
        const std::size_t width = lead <= 0x7FU ? 1U : 2U;
        symbols.emplace_back(piece.substr(index, width));
        index += width;
    }

    std::string key;
    std::vector<std::string> merged;

    while (symbols.size() > 1) {
        std::size_t best_rank = std::numeric_limits<std::size_t>::max();
        std::size_t best_index = 0;
        bool found = false;

        for (std::size_t index = 0;
             index + 1 < symbols.size();
             ++index) {
            key.clear();
            key.append(symbols[index]);
            key.push_back(' ');
            key.append(symbols[index + 1]);

            const auto rank = merge_ranks_m.find(key);
            if (rank != merge_ranks_m.end() &&
                rank->second < best_rank) {
                best_rank = rank->second;
                best_index = index;
                found = true;
            }
        }

        if (!found) {
            break;
        }

        const std::string left = symbols[best_index];
        const std::string right = symbols[best_index + 1];

        merged.clear();
        merged.reserve(symbols.size());
        for (std::size_t index = 0; index < symbols.size();) {
            if (index + 1 < symbols.size() &&
                symbols[index] == left &&
                symbols[index + 1] == right) {
                merged.push_back(left + right);
                index += 2;
            } else {
                merged.push_back(symbols[index]);
                ++index;
            }
        }

        symbols.swap(merged);
    }

    return symbols;
}

void Gpt2Tokenizer::encode_ordinary_text(
    std::string_view text,
    std::vector<TokenId>& token_ids
) const {
    if (text.empty()) {
        return;
    }

    const DecodedText decoded = decode_utf8(text, "encoded text");
    const ByteAlphabet& alphabet = byte_alphabet();
    std::string mapped;

    std::size_t index = 0;
    while (index < decoded.code_points.size()) {
        const std::size_t end = match_piece(decoded.code_points, index);

        mapped.clear();
        const std::size_t first_byte = decoded.byte_offsets[index];
        const std::size_t last_byte = decoded.byte_offsets[end];
        for (std::size_t byte_index = first_byte;
             byte_index < last_byte;
             ++byte_index) {
            const auto byte =
                static_cast<unsigned char>(text[byte_index]);
            append_utf8(mapped, alphabet.byte_to_code_point[byte]);
        }

        for (const std::string& symbol : apply_bpe(mapped)) {
            const auto position = token_to_id_m.find(symbol);
            if (position == token_to_id_m.end()) {
                throw std::runtime_error(
                    "byte-level BPE produced a token outside the "
                    "vocabulary"
                );
            }
            token_ids.push_back(position->second);
        }

        index = end;
    }
}

std::vector<Gpt2Tokenizer::TokenId> Gpt2Tokenizer::encode(
    std::string_view text
) const {
    std::vector<TokenId> token_ids;
    std::size_t position = 0;

    while (position < text.size()) {
        const std::size_t special =
            text.find(end_of_text_token, position);
        if (special == std::string_view::npos) {
            break;
        }

        encode_ordinary_text(
            text.substr(position, special - position),
            token_ids
        );
        token_ids.push_back(end_of_text_id_m);
        position = special + end_of_text_token.size();
    }

    encode_ordinary_text(text.substr(position), token_ids);
    return token_ids;
}

std::string Gpt2Tokenizer::decode(
    std::span<const TokenId> token_ids
) const {
    std::string mapped;
    for (const TokenId token_id : token_ids) {
        if (token_id >= id_to_token_m.size()) {
            throw std::out_of_range(
                "token ID is outside the vocabulary: " +
                std::to_string(token_id)
            );
        }
        mapped.append(id_to_token_m[token_id]);
    }

    const DecodedText decoded = decode_utf8(mapped, "decoded tokens");
    const ByteAlphabet& alphabet = byte_alphabet();
    std::string text;
    text.reserve(decoded.code_points.size());

    for (const char32_t code_point : decoded.code_points) {
        const std::int16_t byte =
            code_point < code_point_limit
                ? alphabet.code_point_to_byte[code_point]
                : static_cast<std::int16_t>(-1);
        if (byte < 0) {
            throw std::logic_error(
                "a vocabulary token left the byte-level alphabet"
            );
        }

        text.push_back(static_cast<char>(
            static_cast<unsigned char>(byte)
        ));
    }

    return text;
}

}  // namespace gpt2
