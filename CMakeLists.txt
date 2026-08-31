cmake_minimum_required(VERSION 3.20)

project(gpt2_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(gpt2
    src/main.cpp
)

target_include_directories(gpt2 PRIVATE include)

include(CTest)

if(BUILD_TESTING)
    add_test(NAME cli_smoke COMMAND gpt2)
    set_tests_properties(cli_smoke PROPERTIES
        PASS_REGULAR_EXPRESSION "GPT-2 CPP inference engine"
    )
endif()
