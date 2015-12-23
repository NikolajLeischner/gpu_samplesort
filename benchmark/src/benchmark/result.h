#pragma once

#include <algorithm>

namespace Benchmark {

	struct Result {
		const double time;
		const std::size_t size;

		Result(double time, std::size_t size) : time(time), size(size) {}
	};

}