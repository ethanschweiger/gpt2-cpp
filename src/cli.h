#pragma once

#include <iosfwd>

namespace gpt2 {

int run_cli(
    int argument_count,
    char* arguments[],
    std::ostream& output,
    std::ostream& error
);

}  // namespace gpt2
