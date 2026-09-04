#include "cli.h"

#include <iostream>

int main(int argument_count, char* arguments[]) {
    return gpt2::run_cli(
        argument_count,
        arguments,
        std::cout,
        std::cerr
    );
}
