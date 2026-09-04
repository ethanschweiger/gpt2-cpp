cmake_minimum_required(VERSION 3.20)

project(gpt2_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(GPT2_WARNINGS_AS_ERRORS
    "Treat compiler warnings as errors"
    OFF
)

option(GPT2_ENABLE_HF_TESTS
    "Enable offline Hugging Face integration tests"
    OFF
)

option(GPT2_ENABLE_GPT2_SMALL_PARITY
    "Enable offline GPT-2 Small full-forward parity testing"
    OFF
)

set(GPT2_SMALL_CHECKPOINT
    "${PROJECT_SOURCE_DIR}/models/gpt2-small-fp32.bin"
    CACHE FILEPATH
    "Path to the exported GPT-2 Small FP32 checkpoint"
)

set(GPT2_SMALL_HF_REFERENCE
    "${PROJECT_SOURCE_DIR}/models/huggingface-cache/hub/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"
    CACHE PATH
    "Path to the matching Hugging Face GPT-2 Small snapshot"
)

function(gpt2_enable_warnings target_name)
    if(MSVC)
        target_compile_options(${target_name} PRIVATE
            /W4
            /permissive-
        )

        if(GPT2_WARNINGS_AS_ERRORS)
            target_compile_options(${target_name} PRIVATE /WX)
        endif()
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "^(AppleClang|Clang|GNU)$")
        target_compile_options(${target_name} PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wsign-conversion
        )

        if(GPT2_WARNINGS_AS_ERRORS)
            target_compile_options(${target_name} PRIVATE -Werror)
        endif()
    endif()
endfunction()

add_library(gpt2_core STATIC
    src/attention.cpp
    src/checkpoint.cpp
    src/layers.cpp
    src/model.cpp
    src/tensor.cpp
    src/tensor_ops.cpp
    src/transformer.cpp
)

target_include_directories(gpt2_core PUBLIC include)
gpt2_enable_warnings(gpt2_core)

add_executable(gpt2
    src/main.cpp
)

target_link_libraries(gpt2 PRIVATE gpt2_core)
gpt2_enable_warnings(gpt2)

include(CTest)

if(BUILD_TESTING)
    find_package(Python3 3.10 COMPONENTS Interpreter REQUIRED)

    add_executable(gpt2_tensor_tests
        tests/tensor_test.cpp
    )

    target_link_libraries(gpt2_tensor_tests PRIVATE gpt2_core)
    gpt2_enable_warnings(gpt2_tensor_tests)

    add_test(NAME tensor_unit COMMAND gpt2_tensor_tests)
    set_tests_properties(tensor_unit PROPERTIES LABELS unit)

    add_executable(gpt2_tensor_ops_tests
        tests/tensor_ops_test.cpp
    )

    target_link_libraries(gpt2_tensor_ops_tests PRIVATE gpt2_core)
    gpt2_enable_warnings(gpt2_tensor_ops_tests)

    add_test(NAME tensor_ops_unit COMMAND gpt2_tensor_ops_tests)
    set_tests_properties(tensor_ops_unit PROPERTIES LABELS unit)

    add_executable(gpt2_checkpoint_tests
        tests/checkpoint_test.cpp
    )

    target_link_libraries(gpt2_checkpoint_tests PRIVATE gpt2_core)
    gpt2_enable_warnings(gpt2_checkpoint_tests)

    add_test(NAME checkpoint_unit COMMAND gpt2_checkpoint_tests)
    set_tests_properties(checkpoint_unit PROPERTIES LABELS unit)

    add_executable(gpt2_layers_tests
        tests/layers_test.cpp
    )

    target_link_libraries(gpt2_layers_tests PRIVATE gpt2_core)
    gpt2_enable_warnings(gpt2_layers_tests)

    add_test(NAME layers_unit COMMAND gpt2_layers_tests)
    set_tests_properties(layers_unit PROPERTIES LABELS unit)

    add_executable(gpt2_attention_tests
        tests/attention_test.cpp
    )

    target_link_libraries(gpt2_attention_tests PRIVATE gpt2_core)
    gpt2_enable_warnings(gpt2_attention_tests)

    add_test(NAME attention_unit COMMAND gpt2_attention_tests)
    set_tests_properties(attention_unit PROPERTIES LABELS unit)

    add_executable(gpt2_transformer_tests
        tests/transformer_test.cpp
    )

    target_link_libraries(gpt2_transformer_tests PRIVATE gpt2_core)
    gpt2_enable_warnings(gpt2_transformer_tests)

    add_test(NAME transformer_unit COMMAND gpt2_transformer_tests)
    set_tests_properties(transformer_unit PROPERTIES LABELS unit)

    add_executable(gpt2_model_tests
        tests/model_test.cpp
    )

    target_link_libraries(gpt2_model_tests PRIVATE gpt2_core)
    gpt2_enable_warnings(gpt2_model_tests)

    add_test(NAME model_unit COMMAND gpt2_model_tests)
    set_tests_properties(model_unit PROPERTIES LABELS unit)

    add_test(
        NAME checkpoint_writer_unit
        COMMAND
            "${Python3_EXECUTABLE}"
            "${PROJECT_SOURCE_DIR}/tests/checkpoint_writer_test.py"
    )
    set_tests_properties(checkpoint_writer_unit PROPERTIES
        LABELS unit
        ENVIRONMENT "PYTHONPATH=${PROJECT_SOURCE_DIR}/tools"
    )

    add_test(
        NAME gpt2_exporter_unit
        COMMAND
            "${Python3_EXECUTABLE}"
            "${PROJECT_SOURCE_DIR}/tests/export_gpt2_test.py"
    )
    set_tests_properties(gpt2_exporter_unit PROPERTIES
        LABELS unit
        ENVIRONMENT "PYTHONPATH=${PROJECT_SOURCE_DIR}/tools"
    )

    if(GPT2_ENABLE_HF_TESTS OR GPT2_ENABLE_GPT2_SMALL_PARITY)
        execute_process(
            COMMAND
                "${Python3_EXECUTABLE}"
                "-c"
                "import numpy, torch, transformers; from transformers import GPT2Config, GPT2LMHeadModel"
            RESULT_VARIABLE GPT2_HF_IMPORT_RESULT
            ERROR_VARIABLE GPT2_HF_IMPORT_ERROR
            OUTPUT_QUIET
        )

        if(NOT GPT2_HF_IMPORT_RESULT EQUAL 0)
            message(FATAL_ERROR
                "Hugging Face tests require packages from "
                "requirements-export.txt:\n${GPT2_HF_IMPORT_ERROR}"
            )
        endif()
    endif()

    if(GPT2_ENABLE_HF_TESTS)
        add_test(
            NAME gpt2_hf_exporter_integration
            COMMAND
                "${Python3_EXECUTABLE}"
                "${PROJECT_SOURCE_DIR}/tests/hf_export_integration_test.py"
        )
        set_tests_properties(gpt2_hf_exporter_integration PROPERTIES
            LABELS "integration;huggingface"
            ENVIRONMENT
                "PYTHONPATH=${PROJECT_SOURCE_DIR}/tools;HF_HUB_OFFLINE=1;TRANSFORMERS_OFFLINE=1"
        )
    endif()

    if(GPT2_ENABLE_GPT2_SMALL_PARITY)
        get_filename_component(
            GPT2_SMALL_CHECKPOINT_ABSOLUTE
            "${GPT2_SMALL_CHECKPOINT}"
            ABSOLUTE
            BASE_DIR "${PROJECT_SOURCE_DIR}"
        )

        get_filename_component(
            GPT2_SMALL_HF_REFERENCE_ABSOLUTE
            "${GPT2_SMALL_HF_REFERENCE}"
            ABSOLUTE
            BASE_DIR "${PROJECT_SOURCE_DIR}"
        )

        if(NOT EXISTS "${GPT2_SMALL_CHECKPOINT_ABSOLUTE}")
            message(FATAL_ERROR
                "GPT-2 Small checkpoint does not exist: "
                "${GPT2_SMALL_CHECKPOINT_ABSOLUTE}"
            )
        endif()

        if(NOT IS_DIRECTORY "${GPT2_SMALL_HF_REFERENCE_ABSOLUTE}")
            message(FATAL_ERROR
                "Hugging Face GPT-2 Small snapshot does not exist: "
                "${GPT2_SMALL_HF_REFERENCE_ABSOLUTE}"
            )
        endif()

        add_executable(gpt2_real_model_runner
            tests/real_model_runner.cpp
        )

        target_link_libraries(gpt2_real_model_runner PRIVATE gpt2_core)
        gpt2_enable_warnings(gpt2_real_model_runner)

        add_test(
            NAME gpt2_real_model_parity
            COMMAND
                "${Python3_EXECUTABLE}"
                "${PROJECT_SOURCE_DIR}/tests/real_model_parity_test.py"
                "--runner"
                "$<TARGET_FILE:gpt2_real_model_runner>"
                "--checkpoint"
                "${GPT2_SMALL_CHECKPOINT_ABSOLUTE}"
                "--hf-reference"
                "${GPT2_SMALL_HF_REFERENCE_ABSOLUTE}"
        )
        set_tests_properties(gpt2_real_model_parity PROPERTIES
            LABELS "integration;huggingface;real-model;slow"
            ENVIRONMENT
                "HF_HUB_OFFLINE=1;TRANSFORMERS_OFFLINE=1;TOKENIZERS_PARALLELISM=false"
            RUN_SERIAL TRUE
            TIMEOUT 900
        )
    endif()

    add_test(NAME cli_smoke COMMAND gpt2)
    set_tests_properties(cli_smoke PROPERTIES
        LABELS smoke
        PASS_REGULAR_EXPRESSION "GPT-2 CPP inference engine"
    )
endif()
