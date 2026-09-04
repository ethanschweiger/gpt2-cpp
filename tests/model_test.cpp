#include "gpt2/model.h"

#include <array>
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
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

using Bytes = std::vector<unsigned char>;

constexpr std::uint32_t vocabulary_size = 7;
constexpr std::uint32_t context_length = 5;
constexpr std::uint32_t embedding_size = 4;
constexpr std::uint32_t head_count = 2;
constexpr std::uint32_t layer_count = 2;

int failure_count = 0;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failure_count;
    }
}

void expect_near(
    float actual,
    float expected,
    float tolerance,
    std::string_view message
) {
    if (!std::isfinite(actual) ||
        std::fabs(actual - expected) > tolerance) {
        std::cerr << "FAIL: " << message
                  << " (expected " << expected
                  << ", got " << actual << ")\n";
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

enum class ValueKind {
    token_embedding,
    weight,
    layer_norm_weight,
    bias,
};

struct TensorFixture {
    std::string name;
    std::vector<std::uint64_t> shape;
    std::vector<float> values;
};

std::size_t element_count(
    const std::vector<std::uint64_t>& shape
) {
    std::size_t result = 1;
    for (const std::uint64_t dimension : shape) {
        result *= static_cast<std::size_t>(dimension);
    }
    return result;
}

float fixture_value(
    std::size_t index,
    std::size_t seed,
    ValueKind kind
) {
    const std::size_t encoded =
        (index * std::size_t{7} + seed * std::size_t{5}) %
        std::size_t{17};
    const int raw = static_cast<int>(encoded) - 8;
    const float value = static_cast<float>(raw);

    switch (kind) {
    case ValueKind::token_embedding:
        return value / 32.0F;
    case ValueKind::weight:
        return value / 64.0F;
    case ValueKind::layer_norm_weight:
        return 1.0F + value / 64.0F;
    case ValueKind::bias:
        return value / 128.0F;
    }

    throw std::logic_error("unknown fixture value kind");
}

TensorFixture make_tensor(
    std::string name,
    std::vector<std::uint64_t> shape,
    ValueKind kind,
    std::size_t seed
) {
    std::vector<float> values(element_count(shape));
    for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] = fixture_value(index, seed, kind);
    }

    return {
        std::move(name),
        std::move(shape),
        std::move(values),
    };
}

void append_model_tensor(
    std::vector<TensorFixture>& tensors,
    std::string name,
    std::vector<std::uint64_t> shape,
    ValueKind kind
) {
    const std::size_t seed = tensors.size() + 1;
    tensors.push_back(make_tensor(
        std::move(name),
        std::move(shape),
        kind,
        seed
    ));
}

std::vector<TensorFixture> make_model_tensors() {
    std::vector<TensorFixture> tensors;
    tensors.reserve(28);

    append_model_tensor(
        tensors,
        "transformer.wte.weight",
        {7, 4},
        ValueKind::token_embedding
    );
    append_model_tensor(
        tensors,
        "transformer.wpe.weight",
        {5, 4},
        ValueKind::weight
    );

    for (std::size_t layer = 0; layer < 2; ++layer) {
        const std::string prefix =
            "transformer.h." + std::to_string(layer);

        append_model_tensor(
            tensors,
            prefix + ".ln_1.weight",
            {4},
            ValueKind::layer_norm_weight
        );
        append_model_tensor(
            tensors,
            prefix + ".ln_1.bias",
            {4},
            ValueKind::bias
        );
        append_model_tensor(
            tensors,
            prefix + ".attn.c_attn.weight",
            {4, 12},
            ValueKind::weight
        );
        append_model_tensor(
            tensors,
            prefix + ".attn.c_attn.bias",
            {12},
            ValueKind::bias
        );
        append_model_tensor(
            tensors,
            prefix + ".attn.c_proj.weight",
            {4, 4},
            ValueKind::weight
        );
        append_model_tensor(
            tensors,
            prefix + ".attn.c_proj.bias",
            {4},
            ValueKind::bias
        );
        append_model_tensor(
            tensors,
            prefix + ".ln_2.weight",
            {4},
            ValueKind::layer_norm_weight
        );
        append_model_tensor(
            tensors,
            prefix + ".ln_2.bias",
            {4},
            ValueKind::bias
        );
        append_model_tensor(
            tensors,
            prefix + ".mlp.c_fc.weight",
            {4, 16},
            ValueKind::weight
        );
        append_model_tensor(
            tensors,
            prefix + ".mlp.c_fc.bias",
            {16},
            ValueKind::bias
        );
        append_model_tensor(
            tensors,
            prefix + ".mlp.c_proj.weight",
            {16, 4},
            ValueKind::weight
        );
        append_model_tensor(
            tensors,
            prefix + ".mlp.c_proj.bias",
            {4},
            ValueKind::bias
        );
    }

    append_model_tensor(
        tensors,
        "transformer.ln_f.weight",
        {4},
        ValueKind::layer_norm_weight
    );
    append_model_tensor(
        tensors,
        "transformer.ln_f.bias",
        {4},
        ValueKind::bias
    );

    return tensors;
}

TensorFixture& find_tensor(
    std::vector<TensorFixture>& tensors,
    std::string_view name
) {
    for (TensorFixture& tensor : tensors) {
        if (tensor.name == name) {
            return tensor;
        }
    }

    throw std::logic_error("fixture tensor not found");
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

Bytes make_checkpoint(const std::vector<TensorFixture>& tensors) {
    Bytes bytes{
        'G', 'P', 'T', '2', 'C', 'P', 'P', '\0'
    };
    append_u32(bytes, 1);
    append_u32(bytes, 64);
    append_u32(bytes, 0x01020304U);
    append_u32(bytes, 0);
    append_u32(bytes, static_cast<std::uint32_t>(tensors.size()));
    append_u32(bytes, vocabulary_size);
    append_u32(bytes, context_length);
    append_u32(bytes, embedding_size);
    append_u32(bytes, head_count);
    append_u32(bytes, layer_count);
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

std::filesystem::path unique_temporary_path() {
    static std::uint64_t counter = 0;
    const auto timestamp = std::chrono::steady_clock::now()
        .time_since_epoch()
        .count();
    return std::filesystem::temp_directory_path() /
        ("gpt2-model-test-" + std::to_string(timestamp) +
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
                "could not write temporary model checkpoint"
            );
        }
    }

    TemporaryCheckpoint(const TemporaryCheckpoint&) = delete;
    TemporaryCheckpoint& operator=(const TemporaryCheckpoint&) = delete;

    ~TemporaryCheckpoint() {
        remove();
    }

    const std::filesystem::path& path() const {
        return path_m;
    }

    void remove() {
        std::error_code error;
        std::filesystem::remove(path_m, error);
    }

private:
    std::filesystem::path path_m;
};

gpt2::Gpt2Model load_model(
    const std::vector<TensorFixture>& tensors
) {
    const TemporaryCheckpoint file(make_checkpoint(tensors));
    return gpt2::Gpt2Model(gpt2::load_checkpoint(file.path()));
}

void test_complete_forward_matches_hugging_face() {
    TemporaryCheckpoint file(
        make_checkpoint(make_model_tensors())
    );
    gpt2::Gpt2Model model(gpt2::load_checkpoint(file.path()));

    expect(
        model.config().vocab_size == vocabulary_size,
        "model exposes its checkpoint configuration"
    );

    file.remove();
    expect(
        !std::filesystem::exists(file.path()),
        "test checkpoint is removed before inference"
    );

    const std::array<std::size_t, 3> token_ids{2, 5, 1};
    const gpt2::Tensor logits = model.forward(token_ids);

    expect(
        logits.shape() == gpt2::Tensor::Shape{3, 7},
        "complete model returns one vocabulary row per input token"
    );

    const std::array<float, 21> expected{
        -0.226207554F, 0.135856345F, 0.598779023F,
        -0.226193503F, 0.135870367F, -0.226263613F,
        -0.226179481F,

        0.510821521F, -0.539808154F, -0.233919472F,
        0.520559788F, -0.530069888F, 0.471868217F,
        0.530298114F,

        -0.366638601F, 0.665413082F, 0.047376595F,
        -0.374848485F, 0.657203197F, -0.333799064F,
        -0.383058369F,
    };

    for (std::size_t index = 0; index < expected.size(); ++index) {
        expect_near(
            logits.at(index),
            expected[index],
            2.0e-5F,
            "complete model matches Hugging Face FP32 logits"
        );
    }
}

void test_model_rejects_invalid_checkpoint_schema() {
    std::vector<TensorFixture> renamed = make_model_tensors();
    find_tensor(
        renamed,
        "transformer.h.0.ln_2.bias"
    ).name = "transformer.h.0.missing.weight";
    expect_throws<std::invalid_argument>(
        [&renamed] {
            static_cast<void>(load_model(renamed));
        },
        "model rejects a missing required tensor"
    );

    std::vector<TensorFixture> unexpected = make_model_tensors();
    unexpected.push_back(make_tensor(
        "lm_head.weight",
        {7, 4},
        ValueKind::weight,
        29
    ));
    expect_throws<std::invalid_argument>(
        [&unexpected] {
            static_cast<void>(load_model(unexpected));
        },
        "model rejects an unexpected language-model head tensor"
    );

    std::vector<TensorFixture> transposed_qkv = make_model_tensors();
    find_tensor(
        transposed_qkv,
        "transformer.h.0.attn.c_attn.weight"
    ).shape = {12, 4};
    expect_throws<std::invalid_argument>(
        [&transposed_qkv] {
            static_cast<void>(load_model(transposed_qkv));
        },
        "model rejects a QKV matrix with transposed dimensions"
    );
}

void test_model_rejects_invalid_token_sequences() {
    gpt2::Gpt2Model model = load_model(make_model_tensors());

    const std::array<std::size_t, 5> full_context{0, 1, 2, 3, 4};
    const gpt2::Tensor full_context_logits = model.forward(full_context);
    expect(
        full_context_logits.shape() == gpt2::Tensor::Shape{5, 7},
        "model accepts a sequence exactly as long as its context window"
    );

    const std::span<const std::size_t> empty;
    expect_throws<std::invalid_argument>(
        [&model, empty] {
            static_cast<void>(model.forward(empty));
        },
        "model rejects an empty token sequence"
    );

    const std::array<std::size_t, 6> too_long{0, 1, 2, 3, 4, 5};
    expect_throws<std::invalid_argument>(
        [&model, &too_long] {
            static_cast<void>(model.forward(too_long));
        },
        "model rejects a sequence longer than its context window"
    );

    const std::array<std::size_t, 1> invalid_token{7};
    expect_throws<std::out_of_range>(
        [&model, &invalid_token] {
            static_cast<void>(model.forward(invalid_token));
        },
        "model rejects a token outside its vocabulary"
    );
}

}  // namespace

int main() {
    test_complete_forward_matches_hugging_face();
    test_model_rejects_invalid_checkpoint_schema();
    test_model_rejects_invalid_token_sequences();

    if (failure_count != 0) {
        std::cerr << failure_count << " model test assertion(s) failed\n";
        return EXIT_FAILURE;
    }

    std::cout << "All model tests passed\n";
    return EXIT_SUCCESS;
}
