#pragma once

namespace SampleSort {
    const int K = 128;
    const int LOG_K = 7;
    __constant__ unsigned long long bst_cache[K];
}
