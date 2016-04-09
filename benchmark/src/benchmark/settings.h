#pragma once

#include <algorithm>
#include <iostream>
#include <math.h>
#include <tclap/CmdLine.h>
#include "distributions.h"
#include "keytype.h"
#include <map>

namespace Benchmark
{
	namespace Algorithm {

		enum class Value { cpu_stl, thrust, samplesort };

		Value parse(std::string value);
		std::string as_string(Value value);

		static const std::map<std::string, Algorithm::Value> types {
			{ "thrust", Algorithm::Value::thrust },
			{ "samplesort", Algorithm::Value::samplesort },
			{ "cpu_stl", Algorithm::Value::cpu_stl } };
	};

struct Settings {

	static Settings parse_from_cmd(int argc, char *argv[], bool disable_exception_handling = false);

	const Algorithm::Value algorithm;
	const Distributions::Type::Value distribution_type;
	const Distributions::Settings distribution_settings;
	const std::uint32_t p;
	const std::uint32_t g;
	const std::uint32_t range;
	const KeyType::Value key_type;
	const bool keys_have_values;
	const std::string output_file;
	const std::vector<std::size_t> sizes;

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