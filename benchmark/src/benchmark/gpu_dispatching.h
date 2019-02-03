#pragma once

#include "settings.h"

namespace Benchmark {
    template<typename KeyType>
    void sort_by_algorithm(Benchmark::Algorithm::Value algorithm, bool keys_only, std::vector<KeyType> &data);
}