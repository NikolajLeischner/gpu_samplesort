#include <bandit/bandit.h>
using namespace bandit;

#include "../src/benchmark/settings.h"
#include "../src/benchmark/distributions.h"
using namespace Benchmark;

go_bandit([]() {
	describe("Benchmark settings", [&]() {

		it("should be created from command line arguments", [&]() {
			int argc(9);
			char *argv[] = { "benchmark", "-a", "samplesort", "-d", "staggered", "-i", "1000", "-i", "128000" };
			auto settings = Settings::parse_from_cmd(argc, argv);

			AssertThat(settings.algorithm, Equals(Algorithm::Value::samplesort));
			AssertThat(settings.distribution_type, Equals(Distributions::Type::Value::staggered));
			AssertThat(settings.sizes.size(), Equals(2));
			AssertThat(settings.sizes, Contains(1000));
			AssertThat(settings.sizes, Contains(128000));
		});

		it("should throw an exception if required arguments are missing", [&]() {
			int argc(3);
			char *argv[] = { "benchmark", "-a", "thrust" };
			AssertThrows(std::exception, Settings::parse_from_cmd(argc, argv, true));
		});

	});
});