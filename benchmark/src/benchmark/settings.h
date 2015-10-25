#pragma once

#include <algorithm>
#include <iostream>
#include <math.h>
#include <distributions.h>
#include <keytype.h>
#include <tclap/CmdLine.h>

namespace Benchmark
{
	struct Algorithm {

		enum class Value { thrust, samplesort };

		static Value parse(std::string value);
	};

struct Settings {

	static Settings parse_from_cmd(int argc, char *argv[]);

	const Algorithm::Value algorithm;
	const Distributions::Type::Value distribution_type;
	const Distributions::Settings distribution_settings;
	const std::uint32_t p;
	const std::uint32_t g;
	const std::uint32_t range;
	const KeyType::Value key_type;
	const bool keys_have_values;
	const std::string output_file;
	const std::vector<std::uint64_t> sizes;

private:

	Settings(Algorithm::Value algorithm,
		Distributions::Type::Value distribution_type,
		Distributions::Settings distribution_settings,
		std::uint32_t p,
		std::uint32_t g,
		std::uint32_t range,
		KeyType::Value key_type,
		bool keys_have_values,
		const std::string& output_file,
		std::vector<std::uint64_t> sizes) :
		algorithm(algorithm),
		distribution_type(distribution_type),
		distribution_settings(distribution_settings),
		p(p),
		g(g),
		range(range),
		key_type(key_type),
		keys_have_values(keys_have_values),
		output_file(output_file),
		sizes(sizes) {}
};

}