#include "gpt2/checkpoint.h"

#include "gpt2/checkpoint_format.h"

#include <algorithm>
#include <array>
#include <bit>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <fstream>
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

constexpr std::uint32_t expected_flags = 0;
constexpr std::uint64_t float32_size = 4;
constexpr std::uint64_t int8_size = 1;
constexpr std::uint64_t tensor_header_size = 32;
constexpr std::uint64_t dimension_size = 8;
constexpr std::uint64_t minimum_name_size = 1;
constexpr std::uint64_t minimum_tensor_record_size =
    tensor_header_size + dimension_size + minimum_name_size + int8_size;
constexpr std::string_view quant_scale_suffix = ".quant_scale";
constexpr std::uint32_t maximum_tensor_count = 4096;
constexpr std::uint32_t maximum_tensor_rank = 8;
constexpr std::uint32_t maximum_tensor_name_size = 1024;
constexpr std::uint64_t maximum_tensor_payload_size =
    std::uint64_t{512} * 1024U * 1024U;
constexpr std::uint64_t maximum_checkpoint_payload_size =
    std::uint64_t{8} * 1024U * 1024U * 1024U;
constexpr std::size_t floats_per_chunk = 4096;

static_assert(CHAR_BIT == 8);
static_assert(sizeof(float) == float32_size);
static_assert(std::numeric_limits<float>::is_iec559);

class BinaryReader {
public:
    explicit BinaryReader(const std::filesystem::path& path)
        : input_m(path, std::ios::binary | std::ios::ate) {
        if (!input_m) {
            throw std::runtime_error(
                "could not open checkpoint: " + path.string()
            );
        }

        const std::ifstream::pos_type end_position = input_m.tellg();
        if (end_position == std::ifstream::pos_type{-1}) {
            throw std::runtime_error(
                "could not determine checkpoint size: " + path.string()
            );
        }

        const std::streamoff file_size =
            static_cast<std::streamoff>(end_position);
        if (file_size < 0) {
            throw std::runtime_error(
                "could not determine checkpoint size: " + path.string()
            );
        }

        const std::uintmax_t unsigned_file_size =
            static_cast<std::uintmax_t>(file_size);
        if (unsigned_file_size >
            std::numeric_limits<std::uint64_t>::max()) {
            throw std::runtime_error("checkpoint file is too large");
        }

        remaining_m = static_cast<std::uint64_t>(unsigned_file_size);
        input_m.seekg(0, std::ios::beg);
        if (!input_m) {
            throw std::runtime_error(
                "could not seek to checkpoint beginning: " + path.string()
            );
        }
    }

    void require_remaining(
        std::uint64_t byte_count,
        std::string_view field_name
    ) const {
        if (byte_count > remaining_m) {
            throw std::runtime_error(
                "checkpoint is truncated while reading " +
                std::string(field_name)
            );
        }
    }

    void read_exact(
        char* destination,
        std::size_t byte_count,
        std::string_view field_name
    ) {
        require_remaining(
            static_cast<std::uint64_t>(byte_count),
            field_name
        );

        if (byte_count >
            static_cast<std::size_t>(
                std::numeric_limits<std::streamsize>::max()
            )) {
            throw std::runtime_error(
                "checkpoint field is too large to read: " +
                std::string(field_name)
            );
        }

        input_m.read(
            destination,
            static_cast<std::streamsize>(byte_count)
        );
        if (!input_m) {
            if (input_m.bad()) {
                throw std::runtime_error(
                    "I/O error while reading checkpoint " +
                    std::string(field_name)
                );
            }

            throw std::runtime_error(
                "checkpoint is truncated while reading " +
                std::string(field_name)
            );
        }

        remaining_m -= static_cast<std::uint64_t>(byte_count);
    }

    std::uint32_t read_u32(std::string_view field_name) {
        std::array<unsigned char, 4> bytes{};
        read_exact(
            reinterpret_cast<char*>(bytes.data()),
            bytes.size(),
            field_name
        );

        return static_cast<std::uint32_t>(bytes[0]) |
            (static_cast<std::uint32_t>(bytes[1]) << 8U) |
            (static_cast<std::uint32_t>(bytes[2]) << 16U) |
            (static_cast<std::uint32_t>(bytes[3]) << 24U);
    }

    std::uint64_t read_u64(std::string_view field_name) {
        std::array<unsigned char, 8> bytes{};
        read_exact(
            reinterpret_cast<char*>(bytes.data()),
            bytes.size(),
            field_name
        );

        std::uint64_t value = 0;
        for (std::size_t index = 0; index < bytes.size(); ++index) {
            value |= static_cast<std::uint64_t>(bytes[index])
                << static_cast<unsigned int>(index * 8U);
        }
        return value;
    }

    std::string read_string(
        std::uint32_t byte_count,
        std::string_view field_name
    ) {
        require_remaining(byte_count, field_name);

        const std::size_t size = static_cast<std::size_t>(byte_count);
        if (size > std::string{}.max_size()) {
            throw std::runtime_error(
                "checkpoint string is too large: " +
                std::string(field_name)
            );
        }

        std::string value(size, '\0');
        read_exact(value.data(), size, field_name);
        return value;
    }

    std::uint64_t remaining() const {
        return remaining_m;
    }

    void require_end_of_file() {
        if (remaining_m != 0) {
            throw std::runtime_error(
                "checkpoint contains trailing bytes"
            );
        }

        const int next_byte = input_m.peek();
        if (next_byte != std::char_traits<char>::eof()) {
            throw std::runtime_error(
                "checkpoint contains trailing bytes"
            );
        }

        if (input_m.bad()) {
            throw std::runtime_error(
                "I/O error while checking checkpoint end"
            );
        }
    }

private:
    std::ifstream input_m;
    std::uint64_t remaining_m = 0;
};

struct GlobalHeader {
    std::uint32_t tensor_count;
    ModelConfig config;
};

struct TensorHeader {
    std::uint32_t name_length;
    std::uint32_t data_type;
    std::uint32_t rank;
    std::uint32_t flags;
    std::uint64_t element_count;
    std::uint64_t payload_size;
};

std::size_t checked_size_t(
    std::uint64_t value,
    std::string_view field_name
) {
    if (value > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error(
            "checkpoint value does not fit in size_t: " +
            std::string(field_name)
        );
    }

    return static_cast<std::size_t>(value);
}

std::uint64_t checked_add(
    std::uint64_t left,
    std::uint64_t right,
    std::string_view field_name
) {
    if (left > std::numeric_limits<std::uint64_t>::max() - right) {
        throw std::runtime_error(
            "checkpoint size overflows uint64: " +
            std::string(field_name)
        );
    }

    return left + right;
}

void require_positive(
    std::uint32_t value,
    std::string_view field_name
) {
    if (value == 0) {
        throw std::runtime_error(
            "checkpoint " + std::string(field_name) +
            " must be greater than zero"
        );
    }
}

GlobalHeader read_global_header(BinaryReader& reader) {
    std::array<char, 8> magic{};
    reader.read_exact(
        magic.data(),
        magic.size(),
        "checkpoint magic"
    );
    if (magic != checkpoint_format::magic) {
        throw std::runtime_error("invalid checkpoint magic");
    }

    const std::uint32_t version = reader.read_u32("format version");
    if (version != checkpoint_format::version) {
        throw std::runtime_error("unsupported checkpoint version");
    }

    const std::uint32_t header_size = reader.read_u32("header size");
    if (header_size != checkpoint_format::header_size) {
        throw std::runtime_error("invalid checkpoint header size");
    }

    const std::uint32_t endian_marker =
        reader.read_u32("endian marker");
    if (endian_marker != checkpoint_format::endian_marker) {
        throw std::runtime_error("invalid checkpoint endian marker");
    }

    const std::uint32_t flags = reader.read_u32("global flags");
    if (flags != expected_flags) {
        throw std::runtime_error("unsupported checkpoint global flags");
    }

    GlobalHeader header{
        reader.read_u32("tensor count"),
        ModelConfig{
            reader.read_u32("vocabulary size"),
            reader.read_u32("context length"),
            reader.read_u32("embedding size"),
            reader.read_u32("attention-head count"),
            reader.read_u32("transformer-layer count")
        }
    };

    for (std::size_t index = 0; index < 4; ++index) {
        if (reader.read_u32("reserved header field") != 0) {
            throw std::runtime_error(
                "checkpoint reserved header fields must be zero"
            );
        }
    }

    require_positive(header.config.vocab_size, "vocabulary size");
    require_positive(header.config.context_length, "context length");
    require_positive(header.config.embedding_size, "embedding size");
    require_positive(header.config.head_count, "attention-head count");
    require_positive(header.config.layer_count, "transformer-layer count");

    if (header.config.embedding_size % header.config.head_count != 0) {
        throw std::runtime_error(
            "checkpoint embedding size must be divisible by head count"
        );
    }

    if (static_cast<std::uint64_t>(header.tensor_count) >
        reader.remaining() / minimum_tensor_record_size) {
        throw std::runtime_error(
            "checkpoint tensor count cannot fit in the remaining file"
        );
    }

    if (header.tensor_count > maximum_tensor_count) {
        throw std::runtime_error(
            "checkpoint contains too many tensor records"
        );
    }

    return header;
}

TensorHeader read_tensor_header(BinaryReader& reader) {
    return TensorHeader{
        reader.read_u32("tensor name length"),
        reader.read_u32("tensor data type"),
        reader.read_u32("tensor rank"),
        reader.read_u32("tensor flags"),
        reader.read_u64("tensor element count"),
        reader.read_u64("tensor payload size")
    };
}

bool is_int8_data_type(std::uint32_t data_type) {
    return data_type ==
        static_cast<std::uint32_t>(checkpoint_format::DataType::int8);
}

bool is_float32_data_type(std::uint32_t data_type) {
    return data_type ==
        static_cast<std::uint32_t>(checkpoint_format::DataType::float32);
}

void validate_tensor_header(
    BinaryReader& reader,
    const TensorHeader& header
) {
    if (!is_float32_data_type(header.data_type) &&
        !is_int8_data_type(header.data_type)) {
        throw std::runtime_error(
            "unsupported checkpoint tensor data type"
        );
    }

    if (header.flags != expected_flags) {
        throw std::runtime_error(
            "unsupported checkpoint tensor flags"
        );
    }

    if (header.rank == 0) {
        throw std::runtime_error(
            "checkpoint tensor rank must be greater than zero"
        );
    }
    if (header.rank > maximum_tensor_rank) {
        throw std::runtime_error(
            "checkpoint tensor rank exceeds the loader limit"
        );
    }

    if (header.name_length == 0) {
        throw std::runtime_error(
            "checkpoint tensor name must not be empty"
        );
    }
    if (header.name_length > maximum_tensor_name_size) {
        throw std::runtime_error(
            "checkpoint tensor name exceeds the loader limit"
        );
    }

    if (header.payload_size > maximum_tensor_payload_size) {
        throw std::runtime_error(
            "checkpoint tensor payload exceeds the loader limit"
        );
    }

    std::uint64_t remaining_record_size =
        static_cast<std::uint64_t>(header.rank) * dimension_size;
    remaining_record_size = checked_add(
        remaining_record_size,
        header.name_length,
        "tensor record"
    );
    remaining_record_size = checked_add(
        remaining_record_size,
        header.payload_size,
        "tensor record"
    );
    reader.require_remaining(remaining_record_size, "tensor record");
}

bool is_valid_utf8(std::string_view value) {
    std::size_t index = 0;

    while (index < value.size()) {
        const auto first = static_cast<unsigned char>(value[index]);
        if (first <= 0x7FU) {
            ++index;
            continue;
        }

        std::size_t continuation_count = 0;
        unsigned char second_minimum = 0x80U;
        unsigned char second_maximum = 0xBFU;

        if (first >= 0xC2U && first <= 0xDFU) {
            continuation_count = 1;
        } else if (first >= 0xE0U && first <= 0xEFU) {
            continuation_count = 2;
            if (first == 0xE0U) {
                second_minimum = 0xA0U;
            } else if (first == 0xEDU) {
                second_maximum = 0x9FU;
            }
        } else if (first >= 0xF0U && first <= 0xF4U) {
            continuation_count = 3;
            if (first == 0xF0U) {
                second_minimum = 0x90U;
            } else if (first == 0xF4U) {
                second_maximum = 0x8FU;
            }
        } else {
            return false;
        }

        if (continuation_count > value.size() - index - 1) {
            return false;
        }

        const auto second =
            static_cast<unsigned char>(value[index + 1]);
        if (second < second_minimum || second > second_maximum) {
            return false;
        }

        for (std::size_t continuation = 2;
             continuation <= continuation_count;
             ++continuation) {
            const auto byte = static_cast<unsigned char>(
                value[index + continuation]
            );
            if (byte < 0x80U || byte > 0xBFU) {
                return false;
            }
        }

        index += continuation_count + 1;
    }

    return true;
}

Tensor::Shape read_shape(
    BinaryReader& reader,
    const TensorHeader& header
) {
    const std::uint64_t dimension_bytes =
        static_cast<std::uint64_t>(header.rank) * dimension_size;
    reader.require_remaining(dimension_bytes, "tensor dimensions");

    const std::size_t rank = checked_size_t(header.rank, "tensor rank");
    Tensor::Shape shape;
    if (rank > shape.max_size()) {
        throw std::runtime_error("checkpoint tensor rank is too large");
    }
    shape.reserve(rank);

    std::uint64_t element_count_u64 = 1;
    std::size_t element_count_size = 1;

    for (std::size_t axis = 0; axis < rank; ++axis) {
        const std::uint64_t dimension =
            reader.read_u64("tensor dimension");
        if (dimension == 0) {
            throw std::runtime_error(
                "checkpoint tensor dimensions must be greater than zero"
            );
        }

        if (element_count_u64 >
            std::numeric_limits<std::uint64_t>::max() / dimension) {
            throw std::runtime_error(
                "checkpoint tensor element count overflows uint64"
            );
        }
        element_count_u64 *= dimension;

        const std::size_t size_dimension =
            checked_size_t(dimension, "tensor dimension");
        if (element_count_size >
            std::numeric_limits<std::size_t>::max() / size_dimension) {
            throw std::runtime_error(
                "checkpoint tensor element count overflows size_t"
            );
        }
        element_count_size *= size_dimension;
        shape.push_back(size_dimension);
    }

    if (element_count_u64 != header.element_count) {
        throw std::runtime_error(
            "checkpoint tensor element count does not match its shape"
        );
    }

    const std::uint64_t bytes_per_element =
        is_int8_data_type(header.data_type) ? int8_size : float32_size;

    if (header.element_count >
        std::numeric_limits<std::uint64_t>::max() / bytes_per_element) {
        throw std::runtime_error(
            "checkpoint tensor payload size overflows uint64"
        );
    }

    const std::uint64_t expected_payload_size =
        header.element_count * bytes_per_element;
    if (header.payload_size != expected_payload_size) {
        throw std::runtime_error(
            "checkpoint tensor payload size does not match its shape"
        );
    }

    return shape;
}

std::string read_tensor_name(
    BinaryReader& reader,
    std::uint32_t name_length
) {
    std::string name = reader.read_string(
        name_length,
        "tensor name"
    );
    if (!is_valid_utf8(name)) {
        throw std::runtime_error(
            "checkpoint tensor name is not valid UTF-8"
        );
    }

    return name;
}

std::uint32_t decode_u32(const unsigned char* bytes) {
    return static_cast<std::uint32_t>(bytes[0]) |
        (static_cast<std::uint32_t>(bytes[1]) << 8U) |
        (static_cast<std::uint32_t>(bytes[2]) << 16U) |
        (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

std::vector<float> read_float32_tensor_values(
    BinaryReader& reader,
    const TensorHeader& header
) {
    reader.require_remaining(header.payload_size, "tensor payload");

    const std::size_t element_count = checked_size_t(
        header.element_count,
        "tensor element count"
    );
    std::vector<float> values;
    if (element_count > values.max_size()) {
        throw std::runtime_error(
            "checkpoint tensor contains too many values"
        );
    }
    values.resize(element_count);

    std::array<unsigned char, floats_per_chunk * float32_size> bytes{};
    std::size_t value_offset = 0;

    while (value_offset < element_count) {
        const std::size_t batch_size = std::min(
            floats_per_chunk,
            element_count - value_offset
        );
        const std::size_t batch_bytes = batch_size * float32_size;
        reader.read_exact(
            reinterpret_cast<char*>(bytes.data()),
            batch_bytes,
            "tensor payload"
        );

        for (std::size_t index = 0; index < batch_size; ++index) {
            const std::uint32_t bits = decode_u32(
                bytes.data() + index * float32_size
            );
            values[value_offset + index] = std::bit_cast<float>(bits);
        }

        value_offset += batch_size;
    }

    return values;
}

std::vector<std::int8_t> read_int8_tensor_values(
    BinaryReader& reader,
    const TensorHeader& header
) {
    reader.require_remaining(header.payload_size, "tensor payload");

    const std::size_t element_count = checked_size_t(
        header.element_count,
        "tensor element count"
    );
    std::vector<std::int8_t> values;
    if (element_count > values.max_size()) {
        throw std::runtime_error(
            "checkpoint tensor contains too many values"
        );
    }
    values.resize(element_count);

    // Each byte already is the value it represents, so no multi-byte
    // decoding is needed the way float32 payloads require.
    reader.read_exact(
        reinterpret_cast<char*>(values.data()),
        element_count,
        "tensor payload"
    );

    for (const std::int8_t value : values) {
        if (value == std::numeric_limits<std::int8_t>::min()) {
            throw std::runtime_error(
                "checkpoint int8 tensor value has no symmetric-"
                "quantization counterpart: -128"
            );
        }
    }

    return values;
}

// Every int8 tensor must be paired with a float32, rank-1 scale
// tensor whose length matches one of its own dimensions. The loader
// enforces the pairing generically; which dimension the scale applies
// to is a property of what the named tensor represents, decided by
// model-loading code rather than by the checkpoint format itself.
void require_quantization_scale(
    const std::string& int8_tensor_name,
    const Int8Tensor& int8_values,
    const std::unordered_map<std::string, Tensor>& tensors
) {
    const std::string scale_name =
        int8_tensor_name + std::string(quant_scale_suffix);
    const auto position = tensors.find(scale_name);
    if (position == tensors.end()) {
        throw std::runtime_error(
            "checkpoint int8 tensor is missing its quantization "
            "scale: " + scale_name
        );
    }

    const Tensor& scale = position->second;
    if (scale.rank() != 1) {
        throw std::runtime_error(
            "checkpoint quantization scale must be a rank-1 tensor: " +
            scale_name
        );
    }

    const Int8Tensor::Shape& shape = int8_values.shape();
    const bool matches_a_dimension =
        std::find(shape.begin(), shape.end(), scale.numel()) !=
        shape.end();
    if (!matches_a_dimension) {
        throw std::runtime_error(
            "checkpoint quantization scale size does not match its "
            "tensor: " + scale_name
        );
    }
}

}  // namespace

Checkpoint::Checkpoint(
    ModelConfig config,
    TensorMap tensors,
    Int8TensorMap int8_tensors
)
    : config_m(config),
      tensors_m(std::move(tensors)),
      int8_tensors_m(std::move(int8_tensors)) {}

const ModelConfig& Checkpoint::config() const {
    return config_m;
}

std::size_t Checkpoint::tensor_count() const {
    return tensors_m.size() + int8_tensors_m.size();
}

bool Checkpoint::contains(std::string_view name) const {
    return tensors_m.find(std::string(name)) != tensors_m.end();
}

const Tensor& Checkpoint::tensor(std::string_view name) const {
    const auto position = tensors_m.find(std::string(name));
    if (position == tensors_m.end()) {
        throw std::out_of_range(
            "checkpoint tensor not found: " + std::string(name)
        );
    }

    return position->second;
}

bool Checkpoint::contains_int8(std::string_view name) const {
    return int8_tensors_m.find(std::string(name)) != int8_tensors_m.end();
}

const Int8Tensor& Checkpoint::int8_tensor(std::string_view name) const {
    const auto position = int8_tensors_m.find(std::string(name));
    if (position == int8_tensors_m.end()) {
        throw std::out_of_range(
            "checkpoint int8 tensor not found: " + std::string(name)
        );
    }

    return position->second;
}

Checkpoint load_checkpoint(const std::filesystem::path& path) {
    BinaryReader reader(path);
    const GlobalHeader header = read_global_header(reader);
    Checkpoint::TensorMap tensors;
    Checkpoint::Int8TensorMap int8_tensors;
    std::uint64_t total_payload_size = 0;

    for (std::uint32_t index = 0;
         index < header.tensor_count;
        ++index) {
        const TensorHeader tensor_header = read_tensor_header(reader);
        validate_tensor_header(reader, tensor_header);

        if (total_payload_size >
            maximum_checkpoint_payload_size - tensor_header.payload_size) {
            throw std::runtime_error(
                "checkpoint payloads exceed the loader limit"
            );
        }
        total_payload_size += tensor_header.payload_size;

        Tensor::Shape shape = read_shape(reader, tensor_header);
        std::string name = read_tensor_name(
            reader,
            tensor_header.name_length
        );

        if (tensors.find(name) != tensors.end() ||
            int8_tensors.find(name) != int8_tensors.end()) {
            throw std::runtime_error(
                "checkpoint contains duplicate tensor: " + name
            );
        }

        if (is_int8_data_type(tensor_header.data_type)) {
            std::vector<std::int8_t> values = read_int8_tensor_values(
                reader,
                tensor_header
            );
            int8_tensors.emplace(
                std::move(name),
                Int8Tensor(std::move(shape), std::move(values))
            );
        } else {
            std::vector<float> values = read_float32_tensor_values(
                reader,
                tensor_header
            );
            tensors.emplace(
                std::move(name),
                Tensor(std::move(shape), std::move(values))
            );
        }
    }

    reader.require_end_of_file();

    // A tensor's scale may be written before or after the tensor
    // itself, so the pairing is only fully checked once every record
    // has been read.
    for (const auto& [name, int8_values] : int8_tensors) {
        require_quantization_scale(name, int8_values, tensors);
    }

    return Checkpoint(
        header.config,
        std::move(tensors),
        std::move(int8_tensors)
    );
}

}  // namespace gpt2
