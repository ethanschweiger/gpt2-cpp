#include "gpt2/checkpoint.h"

#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

using Bytes = std::vector<unsigned char>;

constexpr std::size_t version_offset = 8;
constexpr std::size_t header_size_offset = 12;
constexpr std::size_t endian_marker_offset = 16;
constexpr std::size_t global_flags_offset = 20;
constexpr std::size_t tensor_count_offset = 24;
constexpr std::size_t vocab_size_offset = 28;
constexpr std::size_t context_length_offset = 32;
constexpr std::size_t embedding_size_offset = 36;
constexpr std::size_t head_count_offset = 40;
constexpr std::size_t layer_count_offset = 44;
constexpr std::size_t reserved_offset = 48;

constexpr std::size_t first_tensor_offset = 64;
constexpr std::size_t first_name_length_offset = first_tensor_offset;
constexpr std::size_t first_data_type_offset = first_tensor_offset + 4;
constexpr std::size_t first_rank_offset = first_tensor_offset + 8;
constexpr std::size_t first_flags_offset = first_tensor_offset + 12;
constexpr std::size_t first_element_count_offset = first_tensor_offset + 16;
constexpr std::size_t first_payload_size_offset = first_tensor_offset + 24;
constexpr std::size_t first_dimensions_offset = first_tensor_offset + 32;
constexpr std::size_t first_name_offset = first_dimensions_offset + 16;

int failure_count = 0;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
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

void append_u32(Bytes& bytes, std::uint32_t value) {
    for (unsigned int shift = 0; shift < 32; shift += 8) {
        bytes.push_back(static_cast<unsigned char>(value >> shift));
    }
}

void append_u64(Bytes& bytes, std::uint64_t value) {
    for (unsigned int shift = 0; shift < 64; shift += 8) {
        bytes.push_back(static_cast<unsigned char>(value >> shift));
    }
}

void append_float(Bytes& bytes, float value) {
    append_u32(bytes, std::bit_cast<std::uint32_t>(value));
}

void write_u32_at(
    Bytes& bytes,
    std::size_t offset,
    std::uint32_t value
) {
    for (unsigned int shift = 0; shift < 32; shift += 8) {
        bytes.at(offset + shift / 8) =
            static_cast<unsigned char>(value >> shift);
    }
}

void write_u64_at(
    Bytes& bytes,
    std::size_t offset,
    std::uint64_t value
) {
    for (unsigned int shift = 0; shift < 64; shift += 8) {
        bytes.at(offset + shift / 8) =
            static_cast<unsigned char>(value >> shift);
    }
}

struct TensorFixture {
    std::string name;
    std::vector<std::uint64_t> shape;
    std::vector<float> values;
};

Bytes make_checkpoint(const std::vector<TensorFixture>& tensors) {
    Bytes bytes{
        'G', 'P', 'T', '2', 'C', 'P', 'P', '\0'
    };
    append_u32(bytes, 1);
    append_u32(bytes, 64);
    append_u32(bytes, 0x01020304U);
    append_u32(bytes, 0);
    append_u32(bytes, static_cast<std::uint32_t>(tensors.size()));
    append_u32(bytes, 4);
    append_u32(bytes, 8);
    append_u32(bytes, 4);
    append_u32(bytes, 2);
    append_u32(bytes, 1);
    for (int index = 0; index < 4; ++index) {
        append_u32(bytes, 0);
    }

    for (const TensorFixture& tensor : tensors) {
        append_u32(
            bytes,
            static_cast<std::uint32_t>(tensor.name.size())
        );
        append_u32(bytes, 1);
        append_u32(
            bytes,
            static_cast<std::uint32_t>(tensor.shape.size())
        );
        append_u32(bytes, 0);
        append_u64(
            bytes,
            static_cast<std::uint64_t>(tensor.values.size())
        );
        append_u64(
            bytes,
            static_cast<std::uint64_t>(tensor.values.size()) * 4U
        );

        for (const std::uint64_t dimension : tensor.shape) {
            append_u64(bytes, dimension);
        }
        bytes.insert(bytes.end(), tensor.name.begin(), tensor.name.end());
        for (const float value : tensor.values) {
            append_float(bytes, value);
        }
    }

    return bytes;
}

Bytes make_valid_checkpoint() {
    return make_checkpoint({
        TensorFixture{
            "weight",
            {2, 2},
            {1.0F, -2.5F, -0.0F, 4.0F}
        },
        TensorFixture{
            "beta-\xCE\xB2",
            {2},
            {0.25F, 8.0F}
        }
    });
}

std::filesystem::path unique_temporary_path() {
    static std::uint64_t counter = 0;
    const auto timestamp = std::chrono::steady_clock::now()
        .time_since_epoch()
        .count();
    return std::filesystem::temp_directory_path() /
        ("gpt2-checkpoint-test-" + std::to_string(timestamp) +
         "-" + std::to_string(++counter) + ".bin");
}

class TemporaryCheckpoint {
public:
    explicit TemporaryCheckpoint(const Bytes& bytes)
        : path_m(unique_temporary_path()) {
        std::ofstream output(path_m, std::ios::binary);
        output.write(
            reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size())
        );
        if (!output) {
            throw std::runtime_error(
                "could not write temporary checkpoint"
            );
        }
    }

    TemporaryCheckpoint(const TemporaryCheckpoint&) = delete;
    TemporaryCheckpoint& operator=(const TemporaryCheckpoint&) = delete;

    ~TemporaryCheckpoint() {
        std::error_code error;
        std::filesystem::remove(path_m, error);
    }

    const std::filesystem::path& path() const {
        return path_m;
    }

private:
    std::filesystem::path path_m;
};

void expect_load_failure(const Bytes& bytes, std::string_view message) {
    const TemporaryCheckpoint file(bytes);
    expect_throws<std::runtime_error>(
        [&file] {
            static_cast<void>(gpt2::load_checkpoint(file.path()));
        },
        message
    );
}

void test_valid_checkpoint() {
    const TemporaryCheckpoint file(make_valid_checkpoint());
    const gpt2::Checkpoint checkpoint =
        gpt2::load_checkpoint(file.path());

    const gpt2::ModelConfig& config = checkpoint.config();
    expect(config.vocab_size == 4, "vocabulary size is loaded");
    expect(config.context_length == 8, "context length is loaded");
    expect(config.embedding_size == 4, "embedding size is loaded");
    expect(config.head_count == 2, "head count is loaded");
    expect(config.layer_count == 1, "layer count is loaded");

    expect(checkpoint.tensor_count() == 2, "tensor count is loaded");
    expect(checkpoint.contains("weight"), "known tensor is found");
    expect(!checkpoint.contains("missing"), "unknown tensor is absent");

    const gpt2::Tensor& weight = checkpoint.tensor("weight");
    expect(
        weight.shape() == gpt2::Tensor::Shape{2, 2},
        "matrix shape is loaded"
    );
    expect(weight.at(0) == 1.0F, "positive FP32 value is loaded");
    expect(weight.at(1) == -2.5F, "negative FP32 value is loaded");
    expect(
        weight.at(2) == 0.0F && std::signbit(weight.at(2)),
        "negative zero bit pattern is preserved"
    );
    expect(weight.at(3) == 4.0F, "final matrix value is loaded");

    const gpt2::Tensor& beta = checkpoint.tensor("beta-\xCE\xB2");
    expect(
        beta.shape() == gpt2::Tensor::Shape{2},
        "UTF-8 tensor name and vector shape are loaded"
    );
    expect(beta.at(0) == 0.25F, "vector value is loaded");

    expect_throws<std::out_of_range>(
        [&checkpoint] {
            static_cast<void>(checkpoint.tensor("missing"));
        },
        "missing tensor lookup is rejected"
    );
}

void test_empty_checkpoint() {
    const TemporaryCheckpoint file(make_checkpoint({}));
    const gpt2::Checkpoint checkpoint =
        gpt2::load_checkpoint(file.path());
    expect(
        checkpoint.tensor_count() == 0,
        "the low-level format permits zero tensor records"
    );
}

void test_large_payload_and_special_float_values() {
    std::vector<float> values(4097);
    for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] = static_cast<float>(index);
    }
    values[0] = std::numeric_limits<float>::quiet_NaN();
    values[1] = std::numeric_limits<float>::infinity();
    values[2] = -std::numeric_limits<float>::infinity();

    const TemporaryCheckpoint file(make_checkpoint({
        TensorFixture{"large", {4097}, values}
    }));
    const gpt2::Checkpoint checkpoint =
        gpt2::load_checkpoint(file.path());
    const gpt2::Tensor& loaded = checkpoint.tensor("large");

    expect(std::isnan(loaded.at(0)), "NaN bit pattern is loaded");
    expect(
        std::isinf(loaded.at(1)) && loaded.at(1) > 0.0F,
        "positive infinity bit pattern is loaded"
    );
    expect(
        std::isinf(loaded.at(2)) && loaded.at(2) < 0.0F,
        "negative infinity bit pattern is loaded"
    );
    expect(loaded.at(4095) == 4095.0F, "last value in a chunk is loaded");
    expect(loaded.at(4096) == 4096.0F, "first value after a chunk is loaded");
}

void test_utf8_tensor_names() {
    const std::vector<std::string> valid_names{
        "ascii",
        "beta-\xCE\xB2",
        "euro-\xE2\x82\xAC",
        "brain-\xF0\x9F\xA7\xA0",
        std::string("nul\0name", 8)
    };

    for (const std::string& name : valid_names) {
        const TemporaryCheckpoint file(make_checkpoint({
            TensorFixture{name, {1}, {1.0F}}
        }));
        const gpt2::Checkpoint checkpoint =
            gpt2::load_checkpoint(file.path());
        expect(checkpoint.contains(name), "valid UTF-8 tensor name is loaded");
    }

    const std::vector<std::string> invalid_names{
        std::string("\x80", 1),
        std::string("\xC0\x80", 2),
        std::string("\xE0\x80\x80", 3),
        std::string("\xED\xA0\x80", 3),
        std::string("\xF4\x90\x80\x80", 4),
        std::string("\xF0\x9F", 2)
    };

    for (const std::string& name : invalid_names) {
        expect_load_failure(
            make_checkpoint({TensorFixture{name, {1}, {1.0F}}}),
            "invalid UTF-8 tensor name is rejected"
        );
    }
}

void test_invalid_global_headers() {
    const Bytes valid = make_valid_checkpoint();

    Bytes wrong_magic = valid;
    wrong_magic[0] = 'X';
    expect_load_failure(wrong_magic, "wrong magic is rejected");

    const std::vector<std::pair<std::size_t, std::uint32_t>> cases{
        {version_offset, 2},
        {header_size_offset, 63},
        {endian_marker_offset, 0x04030201U},
        {global_flags_offset, 1},
        {vocab_size_offset, 0},
        {context_length_offset, 0},
        {embedding_size_offset, 0},
        {head_count_offset, 0},
        {layer_count_offset, 0},
        {reserved_offset, 1},
        {reserved_offset + 4, 1},
        {reserved_offset + 8, 1},
        {reserved_offset + 12, 1}
    };

    for (std::size_t index = 0; index < cases.size(); ++index) {
        Bytes malformed = valid;
        write_u32_at(malformed, cases[index].first, cases[index].second);
        expect_load_failure(
            malformed,
            "invalid global header case " + std::to_string(index)
        );
    }

    Bytes indivisible_heads = valid;
    write_u32_at(indivisible_heads, embedding_size_offset, 3);
    expect_load_failure(
        indivisible_heads,
        "embedding size not divisible by head count is rejected"
    );

    Bytes impossible_count = valid;
    write_u32_at(
        impossible_count,
        tensor_count_offset,
        std::numeric_limits<std::uint32_t>::max()
    );
    expect_load_failure(
        impossible_count,
        "impossible tensor count is rejected before allocation"
    );
}

void test_invalid_tensor_metadata() {
    const Bytes valid = make_valid_checkpoint();

    for (const std::uint32_t data_type : {
             0U,
             2U,
             std::numeric_limits<std::uint32_t>::max()
         }) {
        Bytes malformed = valid;
        write_u32_at(malformed, first_data_type_offset, data_type);
        expect_load_failure(
            malformed,
            "unsupported tensor data type is rejected"
        );
    }

    Bytes empty_name = valid;
    write_u32_at(empty_name, first_name_length_offset, 0);
    expect_load_failure(empty_name, "empty tensor name is rejected");

    Bytes zero_rank = valid;
    write_u32_at(zero_rank, first_rank_offset, 0);
    expect_load_failure(zero_rank, "zero tensor rank is rejected");

    Bytes tensor_flags = valid;
    write_u32_at(tensor_flags, first_flags_offset, 1);
    expect_load_failure(tensor_flags, "tensor flags are rejected");

    Bytes zero_dimension = valid;
    write_u64_at(zero_dimension, first_dimensions_offset, 0);
    expect_load_failure(zero_dimension, "zero dimension is rejected");

    Bytes wrong_element_count = valid;
    write_u64_at(wrong_element_count, first_element_count_offset, 5);
    expect_load_failure(
        wrong_element_count,
        "incorrect element count is rejected"
    );

    Bytes wrong_payload_size = valid;
    write_u64_at(wrong_payload_size, first_payload_size_offset, 12);
    expect_load_failure(
        wrong_payload_size,
        "incorrect payload size is rejected"
    );

    Bytes invalid_utf8 = valid;
    invalid_utf8[first_name_offset] = 0xC3U;
    invalid_utf8[first_name_offset + 1] = 0x28U;
    expect_load_failure(invalid_utf8, "invalid UTF-8 name is rejected");

    const Bytes duplicate_names = make_checkpoint({
        TensorFixture{"same", {1}, {1.0F}},
        TensorFixture{"same", {1}, {2.0F}}
    });
    expect_load_failure(duplicate_names, "duplicate tensor names are rejected");
}

void test_overflow_and_allocation_attacks() {
    const Bytes valid = make_valid_checkpoint();

    Bytes huge_rank = valid;
    write_u32_at(
        huge_rank,
        first_rank_offset,
        std::numeric_limits<std::uint32_t>::max()
    );
    expect_load_failure(
        huge_rank,
        "huge rank is rejected before allocation"
    );

    Bytes huge_name = valid;
    write_u32_at(
        huge_name,
        first_name_length_offset,
        std::numeric_limits<std::uint32_t>::max()
    );
    expect_load_failure(
        huge_name,
        "huge name is rejected before allocation"
    );

    Bytes product_overflow = valid;
    write_u64_at(
        product_overflow,
        first_dimensions_offset,
        std::numeric_limits<std::uint64_t>::max()
    );
    write_u64_at(product_overflow, first_dimensions_offset + 8, 2);
    expect_load_failure(
        product_overflow,
        "dimension product overflow is rejected"
    );

    const std::uint64_t payload_overflow_count =
        std::numeric_limits<std::uint64_t>::max() / 4U + 1U;
    Bytes payload_overflow = valid;
    write_u64_at(
        payload_overflow,
        first_dimensions_offset,
        payload_overflow_count
    );
    write_u64_at(payload_overflow, first_dimensions_offset + 8, 1);
    write_u64_at(
        payload_overflow,
        first_element_count_offset,
        payload_overflow_count
    );
    expect_load_failure(
        payload_overflow,
        "payload multiplication overflow is rejected"
    );

    const std::uint64_t huge_element_count =
        std::numeric_limits<std::uint64_t>::max() / 4U;
    Bytes huge_payload = valid;
    write_u64_at(
        huge_payload,
        first_dimensions_offset,
        huge_element_count
    );
    write_u64_at(huge_payload, first_dimensions_offset + 8, 1);
    write_u64_at(
        huge_payload,
        first_element_count_offset,
        huge_element_count
    );
    write_u64_at(
        huge_payload,
        first_payload_size_offset,
        huge_element_count * 4U
    );
    expect_load_failure(
        huge_payload,
        "huge payload is rejected before allocation"
    );

    if constexpr (
        std::numeric_limits<std::size_t>::max() <
        std::numeric_limits<std::uint64_t>::max()
    ) {
        Bytes wide_dimension = valid;
        write_u64_at(
            wide_dimension,
            first_dimensions_offset,
            static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max()
            ) + 1U
        );
        write_u64_at(wide_dimension, first_dimensions_offset + 8, 1);
        expect_load_failure(
            wide_dimension,
            "dimension wider than size_t is rejected"
        );
    }
}

void test_truncation_counts_and_trailing_bytes() {
    const Bytes valid = make_valid_checkpoint();

    for (std::size_t prefix_size = 0;
         prefix_size < valid.size();
         ++prefix_size) {
        const Bytes prefix(
            valid.begin(),
            valid.begin() +
                static_cast<Bytes::difference_type>(prefix_size)
        );
        expect_load_failure(
            prefix,
            "every truncated file prefix is rejected"
        );
    }

    Bytes too_few_records = valid;
    write_u32_at(too_few_records, tensor_count_offset, 1);
    expect_load_failure(
        too_few_records,
        "tensor count smaller than record count is rejected"
    );

    Bytes too_many_records = valid;
    write_u32_at(too_many_records, tensor_count_offset, 3);
    expect_load_failure(
        too_many_records,
        "tensor count larger than record count is rejected"
    );

    Bytes trailing = valid;
    trailing.push_back(0xFFU);
    expect_load_failure(trailing, "trailing byte is rejected");
}

void test_missing_file() {
    const std::filesystem::path missing =
        unique_temporary_path();

    expect_throws<std::runtime_error>(
        [&missing] {
            static_cast<void>(gpt2::load_checkpoint(missing));
        },
        "missing checkpoint file is rejected"
    );
}

}  // namespace

int main() {
    test_valid_checkpoint();
    test_empty_checkpoint();
    test_large_payload_and_special_float_values();
    test_utf8_tensor_names();
    test_invalid_global_headers();
    test_invalid_tensor_metadata();
    test_overflow_and_allocation_attacks();
    test_truncation_counts_and_trailing_bytes();
    test_missing_file();

    if (failure_count != 0) {
        std::cerr << failure_count << " checkpoint test(s) failed\n";
        return EXIT_FAILURE;
    }

    std::cout << "all checkpoint tests passed\n";
    return EXIT_SUCCESS;
}
