#include <bandit/bandit.h>
using namespace bandit;

#include "../src/benchmark/distributions.h"
using namespace Distributions;

namespace {

	void assert_size(Distribution<int> distribution, std::size_t expected_size) {
		it("should have the correct size", [&]() {
			AssertThat(distribution.size(), Equals(expected_size));
		});
	}

	void assert_common_properties(Distribution<int> distribution, Distribution<int> other, std::size_t expected_size) {
		assert_size(distribution, expected_size);

		it("should not be sorted", [&]() {
			AssertThat(std::is_sorted(distribution.begin(), distribution.end()), IsFalse());
		});

		it("should use a constant seed", [&]() {
			AssertThat(distribution.as_vector() == other.as_vector(), IsTrue());
		});
	}
}

go_bandit([](){
  describe("Random distributions", [&]() {

	  std::size_t size(100);
	  std::uint32_t p(8);
	  Settings settings(28, 1);

	  describe("zero distributions", [&]() {
		  auto distribution = zero<int>(size);

		  assert_size(distribution, size);

		  it("should contain only equal elements", [&]() {
			  auto max = std::max_element(distribution.begin(), distribution.end());
			  auto min = std::min_element(distribution.begin(), distribution.end());
			  AssertThat(max, Equals(min));
		  });
	  });

	  describe("sorted distributions", [&]() {
		  auto distribution = sorted<int>(size, settings);

		  assert_size(distribution, size);

		  it("should be sorted", [&]() {
			  AssertThat(std::is_sorted(distribution.begin(), distribution.end()), IsTrue());
		  });
	  });

	  describe("uniform random distributions", [&]() {
		  assert_common_properties(uniform<int>(size, settings), uniform<int>(size, settings), size);
	  });

	  describe("bucket distributions", [&]() {
		  assert_common_properties(bucket<int>(size, settings, p), bucket<int>(size, settings, p), size);
	  });

	  describe("staggered distributions", [&]() {
		  assert_common_properties(staggered<int>(size, settings, p), staggered<int>(size, settings, p), size);
	  });

	  describe("g-groups distributions", [&]() {
		  std::uint32_t g(3);
		  assert_common_properties(g_groups<int>(size, settings, p, g), g_groups<int>(size, settings, p, g), size);
	  });

	  describe("random duplicates distributions", [&]() {
		  std::size_t range(100);
		  assert_common_properties(random_duplicates<int>(size, settings, p, range), random_duplicates<int>(size, settings, p, range), size);
	  });

	  describe("deterministic duplicates distributions", [&]() {
		  std::size_t range(100);
		  assert_common_properties(deterministic_duplicates<int>(size, settings, p), deterministic_duplicates<int>(size, settings, p), size);
	  });
  });
});