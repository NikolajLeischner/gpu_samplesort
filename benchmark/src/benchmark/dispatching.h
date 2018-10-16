#pragma once

#include "settings.h"
#include "result.h"

namespace Benchmark {
    std::vector<Result> execute_with_settings(const Settings &settings);
}