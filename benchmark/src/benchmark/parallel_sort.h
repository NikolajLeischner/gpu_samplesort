#pragma once

#include <algorithm>

#ifdef _MSC_VER
#include <ppl.h>
#elif __GNUC__
#include <parallel/algorithm>
#endif

namespace Benchmark {

    template<class RandomIt>
    void parallel_sort(RandomIt first, RandomIt last) {
#ifdef _MSC_VER
        Concurrency::parallel_sort(first, last);
#elif __GNUC__
        __gnu_parallel::sort(first, last);
#else
        std::sort(first, last);
#endif
    }
}
