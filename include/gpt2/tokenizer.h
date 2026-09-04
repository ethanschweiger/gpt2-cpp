#pragma once

#include <cstddef>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace gpt2 {

class Gpt2Tokenizer {
public:
    using TokenId = std::size_t;

    static Gpt2Tokenizer load(
        const std::filesystem::path& vocabulary_path,
        const std::filesystem::path& merges_path
    );

    std::size_t vocabulary_size() const;

    // The ID of the <|endoftext|> marker, which generation uses as
    // its stopping token.
    TokenId end_of_text_id() const;

    // Splits text on the <|endoftext|> marker, applies the GPT-2
    // pre-tokenizer to the remainder, and byte-level BPE encodes each
    // piece. Throws std::invalid_argument for invalid UTF-8.
    std::vector<TokenId> encode(std::string_view text) const;

    // Concatenates the token strings and reverses the byte-level
    // alphabet. A truncated sequence yields the same incomplete UTF-8
    // bytes the tokens encode.
    std::string decode(
        std::span<const TokenId> token_ids
    ) const;

private:
    using Vocabulary =
        std::unordered_map<std::string, TokenId>;
    using MergeRanks =
        std::unordered_map<std::string, std::size_t>;

    Gpt2Tokenizer(
        Vocabulary token_to_id,
        std::vector<std::string> id_to_token,
        MergeRanks merge_ranks,
        TokenId end_of_text_id
    );

    void encode_ordinary_text(
        std::string_view text,
        std::vector<TokenId>& token_ids
    ) const;

    std::vector<std::string> apply_bpe(
        std::string_view piece
    ) const;

    Vocabulary token_to_id_m;
    std::vector<std::string> id_to_token_m;
    MergeRanks merge_ranks_m;
    TokenId end_of_text_id_m;
};

}  // namespace gpt2
