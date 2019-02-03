#include <iostream>
#include "distributions.h"
#include "dispatching.h"
#include "output.h"
#include "settings.h"
#include <tclap/CmdLine.h>

int main(int argc, const char *argv[]) {
    using namespace Benchmark;
    try {
        auto settings = Settings::parse_from_cmd(argc, argv);

        auto results = execute_with_settings(settings);

        print_results(results, settings.output_file);

        return 0;
    }
    catch (std::exception &e) {
        std::cerr << "Execution failed with exception: " << e.what() << std::endl;
        return 0;
    }
}
