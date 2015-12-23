#pragma once

#include "settings.h"

namespace Benchmark
{
	template<typename KeyType>
	double benchmark_algorithm(Benchmark::Algorithm::Value algorithm, bool keys_have_values, const Distribution<KeyType>& distribution);
}