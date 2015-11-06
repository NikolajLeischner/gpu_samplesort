#include "execution.h"
#include "distributions.h"

namespace Benchmark
{
	namespace {
		using namespace Distributions;

		template<typename KeyType>
		std::vector<double> execute_with_settings(const Settings& settings) {
			std::vector<double> results(settings.sizes.size());
			for (const std::size_t &size : settings.sizes) {
				results.push_back(execute_for_size<KeyType>(settings, size));
			}
			return results;
		}

		template<typename KeyType>
		double execute_for_size(const Settings& settings, std::size_t size) {
			auto distribution = Distributions::create<KeyType>(settings.distribution_type, size,
				settings.distribution_settings, settings.p, settings.g, settings.range);

			return benchmark_algorithm<KeyType>(settings.algorithm, settings.keys_have_values, distribution);
		}

		template<typename KeyType>
		double benchmark_algorithm(Benchmark::Algorithm::Value algorithm, bool keys_have_values, const Distribution<KeyType>& distribution) {
			return 0.0;
		}
	}

	std::vector<double> execute_with_settings(const Settings& settings) {
		switch (settings.key_type) {
		case KeyType::Value::uint64_t:
			return execute_with_settings<std::uint64_t>(settings);
		case KeyType::Value::uint32_t:
			return execute_with_settings<std::uint32_t>(settings);
		case KeyType::Value::uint16_t:
			return execute_with_settings<std::uint16_t>(settings);
		default:
			throw std::exception("Unhandled key type!");
		}
	}
}