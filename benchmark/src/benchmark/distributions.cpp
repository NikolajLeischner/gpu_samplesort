#include "distributions.h"
#include <type_traits>
#include <random>
#include <algorithm>
#include <functional>
#include <limits>
#include <math.h>
#include <numeric>
#include <iostream>

namespace Distributions
{
	namespace {
		template<typename T>
		class Random {
		public:
			Random(const Settings& settings): 
				settings(settings), gen(std::mt19937(rd())), bits_to_remove(sizeof(T) * 8 - settings.bits) {
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
		for (std::size_t i = 0; i < std::min(count, size()); ++i) {
			std::cout << *(begin() + i) << " ";
		}
	}

	template<typename T>
	std::vector<T> Distribution<T>::as_vector() const {
		return content;
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
			if (i < (p / 2)) {
				offset *= (2 * p) + 1;
			}
			else {
				offset *= (static_cast<std::uint32_t>(i) - (p / 2)) * 2;
			}

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

				const T x = ((j * g) + (p / 2) + j) % p;
				const T y = ((j * g) + (p / 2) + j + 1) % p;

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
			const T sumOfRegions = std::accumulate(regions.begin(), regions.end(), 0);
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


	template class Distribution<int>;
	template Distribution<int> zero(std::size_t size);
	template Distribution<int> sorted(std::size_t size, const Settings& settings);
	template Distribution<int> sorted_descending(std::size_t size, const Settings& settings);
	template Distribution<int> uniform(std::size_t size, const Settings& settings);
	template Distribution<int> gaussian(std::size_t size, const Settings& settings);
	template Distribution<int> bucket(std::size_t size, const Settings& settings, std::uint32_t p);
	template Distribution<int> staggered(std::size_t size, const Settings& settings, std::uint32_t p);
	template Distribution<int> g_groups(std::size_t size, const Settings& settings, std::uint32_t p, std::uint32_t g);
	template Distribution<int> random_duplicates(std::size_t size, const Settings& settings, std::uint32_t p, std::size_t range);
	template Distribution<int> deterministic_duplicates(std::size_t size, const Settings& settings, std::uint32_t p);
}
