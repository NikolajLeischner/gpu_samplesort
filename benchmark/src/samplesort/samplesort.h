#pragma once

#include <algorithm>

namespace SampleSort {
    void sort(std::uint16_t *begin, std::uint16_t *end);

    void sort(std::uint32_t *begin, std::uint32_t *end);

    void sort(std::uint64_t *begin, std::uint64_t *end);

    void sort_by_key(std::uint16_t *begin, std::uint16_t *end, std::uint64_t *values);

    void sort_by_key(std::uint32_t *begin, std::uint32_t *end, std::uint64_t *values);

    void sort_by_key(std::uint64_t *begin, std::uint64_t *end, std::uint64_t *values);
}
