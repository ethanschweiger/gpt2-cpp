#include "gpt2/model.h"

#include "model_fixture.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

using gpt2_test::find_tensor;
using gpt2_test::make_checkpoint;
using gpt2_test::make_model_tensors;
using gpt2_test::make_tensor;
using gpt2_test::QuantizationTarget;
using gpt2_test::quantize_model_tensors;
using gpt2_test::QuantizedFixtures;
using gpt2_test::TemporaryCheckpoint;
using gpt2_test::TensorFixture;
using gpt2_test::tied_embedding_target;
using gpt2_test::transformer_weight_targets;
using gpt2_test::ValueKind;

constexpr std::uint32_t vocabulary_size =
    gpt2_test::fixture_vocabulary_size;

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

// Asserts that a quantized model's forward pass matches an FP32
// model's forward pass computed over the same explicitly dequantized
// weights -- the same cross-check tensor_ops_test.cpp/layers_test.cpp/
// attention_test.cpp/transformer_test.cpp use for the individual
// operations, now over the whole model. See ChannelQuantized's comment
// (model_fixture.h) for why this should hold near-exactly rather than
// only approximately: quantized_matmul dequantizes each weight with a
// single `static_cast<float>(int8) * scale` before the same summation
// loop matmul uses, so an operation built from these should reproduce
// its FP32 sibling to within ordinary floating-point slack, not
// quantization error.
void expect_forward_matches_dequantized_reference(
    const QuantizedFixtures& fixtures,
    std::string_view message
) {
    const gpt2::Gpt2Model quantized_model = load_model(fixtures.quantized);
    const gpt2::Gpt2Model reference_model =
        load_model(fixtures.dequantized_reference);

    const std::array<std::size_t, 3> token_ids{2, 5, 1};
    const gpt2::Tensor quantized_logits =
        quantized_model.forward(token_ids);
    const gpt2::Tensor reference_logits =
        reference_model.forward(token_ids);

    expect(
        quantized_logits.shape() == reference_logits.shape(),
        message
    );
    for (std::size_t index = 0; index < quantized_logits.numel(); ++index) {
        expect_near(
            quantized_logits.at(index),
            reference_logits.at(index),
            1.0e-4F,
            message
        );
    }
}

void test_quantized_transformer_weights_match_dequantized_reference() {
    const QuantizedFixtures fixtures =
        quantize_model_tensors(transformer_weight_targets());

    expect_forward_matches_dequantized_reference(
        fixtures,
        "a transformer-quantized model (FP32 wte) matches the FP32 "
        "model run over the same explicitly dequantized weights"
    );
}

void test_fully_quantized_model_matches_dequantized_reference() {
    std::vector<QuantizationTarget> targets = transformer_weight_targets();
    targets.push_back(tied_embedding_target());
    const QuantizedFixtures fixtures = quantize_model_tensors(targets);

    expect_forward_matches_dequantized_reference(
        fixtures,
        "a fully-quantized model (transformer weights and wte) matches "
        "the FP32 model run over the same explicitly dequantized "
        "weights"
    );
}

void test_quantized_only_tied_embedding_matches_dequantized_reference() {
    const QuantizedFixtures fixtures =
        quantize_model_tensors({tied_embedding_target()});

    expect_forward_matches_dequantized_reference(
        fixtures,
        "a model quantizing only wte (FP32 transformer layers) matches "
        "the FP32 model run over the same explicitly dequantized "
        "weights"
    );
}

void test_model_rejects_mixed_precision_within_a_layer() {
    // Quantize only c_attn in layer 0; every other linear weight in
    // that layer stays FP32. Neither transformer_block nor
    // quantized_transformer_block can run a layer like this.
    const QuantizedFixtures fixtures = quantize_model_tensors({
        {"transformer.h.0.attn.c_attn.weight", 1},
    });

    expect_throws<std::invalid_argument>(
        [&fixtures] {
            static_cast<void>(load_model(fixtures.quantized));
        },
        "model rejects a layer that mixes float32 and int8 weights"
    );
}

void test_model_rejects_inconsistent_layer_precision() {
    // Layer 0 is fully quantized and internally consistent; layer 1 is
    // left entirely FP32. tools/export_gpt2.py's --quantize flag
    // applies to every transformer layer uniformly, so a checkpoint
    // like this cannot come from that exporter.
    std::vector<QuantizationTarget> targets{
        {"transformer.h.0.attn.c_attn.weight", 1},
        {"transformer.h.0.attn.c_proj.weight", 1},
        {"transformer.h.0.mlp.c_fc.weight", 1},
        {"transformer.h.0.mlp.c_proj.weight", 1},
    };
    const QuantizedFixtures fixtures = quantize_model_tensors(targets);

    expect_throws<std::invalid_argument>(
        [&fixtures] {
            static_cast<void>(load_model(fixtures.quantized));
        },
        "model rejects transformer layers that disagree on precision"
    );
}

void test_model_rejects_quantization_scale_on_the_wrong_axis() {
    QuantizedFixtures fixtures =
        quantize_model_tensors({tied_embedding_target()});
    TensorFixture& scale = find_tensor(
        fixtures.quantized,
        "transformer.wte.weight.quant_scale"
    );
    expect(
        scale.values.size() == vocabulary_size,
        "the wte scale fixture starts with one entry per vocabulary "
        "word"
    );

    // embedding_size (4) is a genuine dimension of wte's [7, 4] shape,
    // so checkpoint.cpp's own generic "matches one of the tensor's
    // dimensions" check would accept this; only the model's
    // axis-specific check (wte's channel axis is its rows, not its
    // columns -- see docs/quantization.md) should catch it.
    constexpr std::uint32_t embedding_size =
        gpt2_test::fixture_embedding_size;
    scale.shape = {static_cast<std::uint64_t>(embedding_size)};
    scale.values.assign(embedding_size, 1.0F);

    expect_throws<std::invalid_argument>(
        [&fixtures] {
            static_cast<void>(load_model(fixtures.quantized));
        },
        "model rejects a wte quantization scale sized for the wrong "
        "axis"
    );
}

}  // namespace

int main() {
    test_complete_forward_matches_hugging_face();
    test_model_rejects_invalid_checkpoint_schema();
    test_model_rejects_invalid_token_sequences();
    test_quantized_transformer_weights_match_dequantized_reference();
    test_fully_quantized_model_matches_dequantized_reference();
    test_quantized_only_tied_embedding_matches_dequantized_reference();
    test_model_rejects_mixed_precision_within_a_layer();
    test_model_rejects_inconsistent_layer_precision();
    test_model_rejects_quantization_scale_on_the_wrong_axis();

    if (failure_count != 0) {
        std::cerr << failure_count << " model test assertion(s) failed\n";
        return EXIT_FAILURE;
    }

    std::cout << "All model tests passed\n";
    return EXIT_SUCCESS;
}
