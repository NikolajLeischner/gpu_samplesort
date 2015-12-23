#pragma once

#include "settings.h"
#include "result.h"

namespace Benchmark
{
	void print_results(const std::vector<Result>& results, const std::string& output_file);
}