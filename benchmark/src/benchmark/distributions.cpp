#include "distributions.h"
#include <type_traits>
#include <random>
#include <algorithm>
#include <functional>
#include <limits>
#include <math.h>
#include <numeric>
#include <iostream>
#include <map>
#include <utility>

namespace Distributions
{
	namespace {
		template<typename T>
		class Random {
		public:
			Random(const Settings& settings): settings(settings), gen(std::mt19937(rd())), 
				bits_to_remove(sizeof(T) * 8 - settings.bits) {
				gen.seed(17);
			}

			T operator()() { 
				T result = distribution(gen);
				for (std::size_t i = 1; i < settings.samples; ++i)
					result &= distribution(gen);
				return result >> bits_to_remove;
			}

		private:
			const Settings settings;
			std::random_device rd;
			std::mt19937 gen;
			const std::uniform_int_distribution<T> distribution;
			const std::size_t bits_to_remove;
		};
	}


	template<typename T>
	Distribution<T>::Distribution(std::vector<T> content): content(content) {
		static_assert(std::is_integral<T>(), "The type must be integral!");
	}

	template<typename T>
	const T* const Distribution<T>::begin() const {
		return content.data();
	}

	template<typename T>
	std::size_t Distribution<T>::size() const {
		return content.size();
	}

	template<typename T>
	const T* const Distribution<T>::end() const {
		return content.data() + size();
	}

	template<typename T>
	std::size_t Distribution<T>::memory_size() const {
		return sizeof(T) * size();
	}

	template<typename T>
	void Distribution<T>::print(std::size_t count) const {
		for (std::size_t i = 0; i < std::min(count, size()); ++i)
			std::cout << *(begin() + i) << " ";
	}

	template<typename T>
	std::vector<T> Distribution<T>::as_vector() const {
		return content;
	}

	Type::Value Type::parse(std::string value) {

		std::transform(value.begin(), value.end(), value.begin(), tolower);

		std::map<std::string, Type::Value> types{
			{"zero", Type::Value::zero},
			{"sorted", Type::Value::sorted},
			{"uniform", Type::Value::uniform},
			{"gaussian", Type::Value::gaussian},
			{"bucket", Type::Value::bucket},
			{"staggered", Type::Value::staggered},
			{"g-groups", Type::Value::g_groups},
			{"sorted-descending", Type::Value::sorted_descending},
			{"random-duplicates", Type::Value::random_duplicates},
			{"deterministic-duplicates", Type::Value::deterministic_duplicates} };

		auto result = types.find(value);
		if (result == types.end())
			throw new std::exception();
		else
			return result->second;
	}

	template<typename T>
	Distribution<T> create(const Type::Value type, std::size_t size, const Settings& settings, std::uint32_t p, std::uint32_t g, std::uint32_t range) {
		switch (type) {
		case Type::Value::sorted:
			return sorted<T>(size, settings);
		case Type::Value::uniform:
			return uniform<T>(size, settings);
		case Type::Value::gaussian:
			return gaussian<T>(size, settings);
		case Type::Value::bucket:
			return bucket<T>(size, settings, p);
		case Type::Value::g_groups:
			return g_groups<T>(size, settings, p, g);
		case Type::Value::sorted_descending:
			return sorted_descending<T>(size, settings);
		case Type::Value::staggered:
			return staggered<T>(size, settings, p);
		case Type::Value::deterministic_duplicates:
			return deterministic_duplicates<T>(size, settings, p);
		case Type::Value::random_duplicates:
			return random_duplicates<T>(size, settings, p, range);
		case Type::Value::zero:
			return zero<T>(size);
		}
	}

	template<typename T>
	Distribution<T> zero(std::size_t size) {
		std::vector<T> content(size);
		std::fill(content.begin(), content.end(), 0);
		return Distribution<T>(content);
	}


	template<typename T>
	Distribution<T> sorted(std::size_t size, const Settings& settings) {
		std::vector<T> content(size);
		Random<T> random(settings);
		std::generate(content.begin(), content.end(), [&]() { return random(); });
		std::sort(content.begin(), content.end());
		return Distribution<T>(content);
	}

	template<typename T>
	Distribution<T> sorted_descending(std::size_t size, const Settings& settings) {
		std::vector<T> content(size);
		Random<T> random(settings);
		std::generate(content.begin(), content.end(), [&]() { return random(); });
		std::sort(content.begin(), content.end(), std::greater<T>());
		return Distribution<T>(content);
	}


	template<typename T>
	Distribution<T> uniform(std::size_t size, const Settings& settings) {
		std::vector<T> content(size);
		Random<T> random(settings);
		std::generate(content.begin(), content.end(), [&]() { return random(); });
		return Distribution<T>(content);
	}

	template<typename T>
	Distribution<T> gaussian(std::size_t size, const Settings& settings) {
		std::vector<T> content(size);
		Random<T> random(settings);
		std::generate(content.begin(), content.end(), [&]() { return (random() + random() + random() + random()) / 4; });
		return Distribution<T>(content);
	}

	template<typename T>
	Distribution<T> bucket(std::size_t size, const Settings& settings, std::uint32_t p) {
		std::vector<T> content(size);
		Random<T> random(settings);

		auto iterator = content.begin();
		const T offset = std::numeric_limits<T>::max() / (p + 1);

		for (std::size_t i = 0; i < p; ++i) {
			for (std::size_t j = 0; j < p; ++j) {
				auto end = iterator + (size / (p * p));
				std::generate(iterator, end, [&]() { return (static_cast<T>(j) * offset) + (random() % offset); });
				iterator = end;
			}
		}

		std::generate(iterator, content.end(), [&]() { return random(); });

		return Distribution<T>(content);
	}

	template<typename T>
	Distribution<T> staggered(std::size_t size, const Settings& settings, std::uint32_t p) {
		std::vector<T> content(size);
		Random<T> random(settings);

		auto iterator = content.begin();

		for (std::size_t i = 0; i < p; ++i) {
			T offset = std::numeric_limits<T>::max() / (p + 1);
			if (i < (p / 2))
				offset *= (2 * p) + 1;
			else
				offset *= (static_cast<std::uint32_t>(i) - (p / 2)) * 2;

			auto end = iterator + (size / p);
			std::generate(iterator, end, [&]() { return offset + (random() / p) + 1; });
			iterator = end;
		}

		std::generate(iterator, content.end(), [&]() { return random(); });

		return Distribution<T>(content);
	}

	template<typename T>
	Distribution<T> g_groups(std::size_t size, const Settings& settings, std::uint32_t p, std::uint32_t g) {
		std::vector<T> content(size);
		Random<T> random(settings);

		auto iterator = content.begin();
		const T offset = std::numeric_limits<T>::max() / (p + 1);

		for (std::size_t i = 0; i < p; ++i) {
			for (std::size_t j = 0; j < g; ++j) {

				const T x = static_cast<T>(((j * g) + (p / 2) + j) % p);
				const T y = static_cast<T>(((j * g) + (p / 2) + j + 1) % p);

				auto end = iterator + (size / (p * g));
				std::generate(iterator, end, [&]() { return offset * x + random() % (offset * (y - x) - 1); });
				iterator = end;
			}
		}

		std::generate(iterator, content.end(), [&]() { return random(); });

		return Distribution<T>(content);
	}

	template<typename T>
	Distribution<T> random_duplicates(std::size_t size, const Settings& settings, std::uint32_t p, std::size_t range) {
		std::vector<T> content(size);
		Random<T> random(settings);

		auto iterator = content.begin();
		for (std::size_t i = 0; i < p; ++i) {

			std::vector<T> regions(range);
			std::generate(regions.begin(), regions.end(), [&]() { return random() % static_cast<std::uint32_t>(range); });
			const T sumOfRegions = std::accumulate(regions.begin(), regions.end(), static_cast<T>(0));
			std::transform(regions.begin(), regions.end(), regions.begin(), [&](T value) { 
				return (value * (static_cast<std::uint32_t>(size) / p)) / sumOfRegions; });

			for (std::size_t j = 0; j < range; ++j) {
				T value = random() % range;
				auto end = iterator + regions[j];
				std::fill(iterator, end, value);
				iterator = end;
			}
		}

		std::generate(iterator, content.end(), [&]() { return random(); });

		return Distribution<T>(content);
	}

	template<typename T>
	Distribution<T> deterministic_duplicates(std::size_t size, const Settings& settings, std::uint32_t p) {
		std::vector<T> content(size);

		auto iterator = content.begin();
		for (std::size_t i = 2; i < p; i *= 2) {
			auto end = iterator + (size / i);
			std::fill(iterator, end, (T)(log((double)size * 2 / i) / log(2.0)));
			iterator = end;
		}
		for (std::size_t i = 2; i < p; i *= 2) {
			auto end = iterator + (p / i);
			std::fill(iterator, end, (T)(log((double)size * 2 / i) / log(2.0)));
			iterator = end;
		}

		Random<T> random(settings);
		std::generate(iterator, content.end(), [&]() { return random(); });

		return Distribution<T>(content);
	}

	template class Distribution<std::uint16_t>;
	template class Distribution<std::uint32_t>;
	template class Distribution<std::uint64_t>;

	template Distribution<std::uint16_t> create(Type::Value type, std::size_t size, const Settings& settings, std::uint32_t p, std::uint32_t g, std::uint32_t range);
	template Distribution<std::uint32_t> create(Type::Value type, std::size_t size, const Settings& settings, std::uint32_t p, std::uint32_t g, std::uint32_t range);
	template Distribution<std::uint64_t> create(Type::Value type, std::size_t size, const Settings& settings, std::uint32_t p, std::uint32_t g, std::uint32_t range);
	
	template Distribution<std::uint16_t> zero(std::size_t size);
	template Distribution<std::uint16_t> sorted(std::size_t size, const Settings& settings);
	template Distribution<std::uint16_t> sorted_descending(std::size_t size, const Settings& settings);
	template Distribution<std::uint16_t> uniform(std::size_t size, const Settings& settings);
	template Distribution<std::uint16_t> gaussian(std::size_t size, const Settings& settings);
	template Distribution<std::uint16_t> bucket(std::size_t size, const Settings& settings, std::uint32_t p);
	template Distribution<std::uint16_t> staggered(std::size_t size, const Settings& settings, std::uint32_t p);
	template Distribution<std::uint16_t> g_groups(std::size_t size, const Settings& settings, std::uint32_t p, std::uint32_t g);
	template Distribution<std::uint16_t> random_duplicates(std::size_t size, const Settings& settings, std::uint32_t p, std::size_t range);
	template Distribution<std::uint16_t> deterministic_duplicates(std::size_t size, const Settings& settings, std::uint32_t p);

	template Distribution<std::uint32_t> zero(std::size_t size);
	template Distribution<std::uint32_t> sorted(std::size_t size, const Settings& settings);
	template Distribution<std::uint32_t> sorted_descending(std::size_t size, const Settings& settings);
	template Distribution<std::uint32_t> uniform(std::size_t size, const Settings& settings);
	template Distribution<std::uint32_t> gaussian(std::size_t size, const Settings& settings);
	template Distribution<std::uint32_t> bucket(std::size_t size, const Settings& settings, std::uint32_t p);
	template Distribution<std::uint32_t> staggered(std::size_t size, const Settings& settings, std::uint32_t p);
	template Distribution<std::uint32_t> g_groups(std::size_t size, const Settings& settings, std::uint32_t p, std::uint32_t g);
	template Distribution<std::uint32_t> random_duplicates(std::size_t size, const Settings& settings, std::uint32_t p, std::size_t range);
	template Distribution<std::uint32_t> deterministic_duplicates(std::size_t size, const Settings& settings, std::uint32_t p);

	template Distribution<std::uint64_t> zero(std::size_t size);
	template Distribution<std::uint64_t> sorted(std::size_t size, const Settings& settings);
	template Distribution<std::uint64_t> sorted_descending(std::size_t size, const Settings& settings);
	template Distribution<std::uint64_t> uniform(std::size_t size, const Settings& settings);
	template Distribution<std::uint64_t> gaussian(std::size_t size, const Settings& settings);
	template Distribution<std::uint64_t> bucket(std::size_t size, const Settings& settings, std::uint32_t p);
	template Distribution<std::uint64_t> staggered(std::size_t size, const Settings& settings, std::uint32_t p);
	template Distribution<std::uint64_t> g_groups(std::size_t size, const Settings& settings, std::uint32_t p, std::uint32_t g);
	template Distribution<std::uint64_t> random_duplicates(std::size_t size, const Settings& settings, std::uint32_t p, std::size_t range);
	template Distribution<std::uint64_t> deterministic_duplicates(std::size_t size, const Settings& settings, std::uint32_t p);
}
