#include <bandit/bandit.h>

using namespace bandit;

#include "../src/benchmark/distributions.h"

using namespace Distributions;

namespace {
    void assert_size(Distribution<std::uint32_t> distribution, std::size_t expected_size) {
        it("should have the correct size", [&]() {
            AssertThat(distribution.size(), Equals(expected_size));
        });
    }

    void assert_common_properties(Distribution<std::uint32_t> distribution, Distribution<std::uint32_t> other,
                                  std::size_t expected_size) {
        assert_size(distribution, expected_size);

        it("should not be sorted", [&]() {
            AssertThat(std::is_sorted(distribution.begin(), distribution.end()), IsFalse());
        });

        it("should use a constant seed", [&]() {
            AssertThat(distribution.as_vector() == other.as_vector(), IsTrue());
        });
    }
}

go_bandit([]() {
    describe("Random distributions", [&]() {

        std::size_t size(1000);
        std::uint32_t p(8);
        Settings settings(28, 1);

        describe("zero distributions", [&]() {
            auto distribution = zero<std::uint32_t>(size);

            assert_size(distribution, size);

            it("should contain only equal elements", [&]() {
                auto max = std::max_element(distribution.begin(), distribution.end());
                auto min = std::min_element(distribution.begin(), distribution.end());
                AssertThat(max, Equals(min));
            });
        });

        describe("sorted distributions", [&]() {
            auto distribution = sorted<std::uint32_t>(size, settings);

            assert_size(distribution, size);

            it("should be sorted", [&]() {
                AssertThat(std::is_sorted(distribution.begin(), distribution.end()), IsTrue());
            });
        });

        describe("descending sorted distributions", [&]() {
            auto distribution = sorted_descending<std::uint32_t>(size, settings);

            assert_size(distribution, size);

            it("should be sorted", [&]() {
                std::vector<std::uint32_t> data = distribution.as_vector();
                AssertThat(std::is_sorted(data.rbegin(), data.rend()), IsTrue());
            });
        });

        describe("uniform random distributions", [&]() {
            assert_common_properties(uniform<std::uint32_t>(size, settings), uniform<std::uint32_t>(size, settings),
                                     size);
        });

        describe("bucket distributions", [&]() {
            assert_common_properties(bucket<std::uint32_t>(size, settings, p), bucket<std::uint32_t>(size, settings, p),
                                     size);
        });

        describe("staggered distributions", [&]() {
            assert_common_properties(staggered<std::uint32_t>(size, settings, p),
                                     staggered<std::uint32_t>(size, settings, p), size);
        });

        describe("g-groups distributions", [&]() {
            std::uint32_t g(3);
            assert_common_properties(g_groups<std::uint32_t>(size, settings, p, g),
                                     g_groups<std::uint32_t>(size, settings, p, g), size);
        });

        describe("random duplicates distributions", [&]() {
            std::size_t range(100);
            assert_common_properties(random_duplicates<std::uint32_t>(size, settings, p, range),
                                     random_duplicates<std::uint32_t>(size, settings, p, range), size);
        });

        describe("deterministic duplicates distributions", [&]() {
            std::size_t range(100);
            assert_common_properties(deterministic_duplicates<std::uint32_t>(size, settings, p),
                                     deterministic_duplicates<std::uint32_t>(size, settings, p), size);
        });
    });
});