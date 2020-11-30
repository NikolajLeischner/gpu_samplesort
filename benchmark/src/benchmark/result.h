#pragma once

#include <algorithm>
#include <ostream>

namespace Benchmark {

    struct Result {
        const double time;
        const std::size_t size;

        void write_to_stream(std::ostream &stream) const {
            stream << size << ";" << time << ";" << std::endl;
        }

        Result(double time, std::size_t size) : time(time), size(size) {}
    };

}
