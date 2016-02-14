#include "output.h"
#include <iostream>
#include <fstream>

namespace Benchmark
{
	void print_results(const std::vector<Result>& results, const std::string& output_file) {
		std::ofstream file(output_file);
		for (auto result : results)
			result.write_to_stream(file);
	}
}