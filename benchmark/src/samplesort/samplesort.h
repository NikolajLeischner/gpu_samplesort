#pragma once
#include <algorithm>


#include "samplesort/detail/Lrand48.inl"
#include "samplesort/detail/CTAScan.inl"
#include "samplesort/detail/CTAOddEven.inl"

namespace SampleSort
{
    template<typename T, typename BinaryPredicate>
    __device__ T min(const T &lhs, const T &rhs, BinaryPredicate comp)
    {
        return comp(lhs, rhs) ? lhs : rhs;
    }

    template<typename T>
    __device__ T min(const T &lhs, const T &rhs)
    {
        return lhs < rhs ? lhs : rhs;
    }

    template<typename T, typename BinaryPredicate>
    __device__ T max(const T &lhs, const T &rhs, BinaryPredicate comp)
    {
        return comp(lhs,rhs) ? rhs : lhs;
    }

    template<typename T>
    __device__ T max(const T &lhs, const T &rhs)
    {
        return lhs < rhs ? rhs : lhs;
    }

    const int K = 128;
    const int LOG_K = 7;
    __constant__ unsigned long long bstCache[K];


  void sort(std::uint16_t *begin, std::uint16_t *end);
  void sort(std::uint32_t *begin, std::uint32_t *end);
  void sort(std::uint64_t *begin, std::uint64_t *end);

  void sort_by_key(std::uint16_t* begin, std::uint16_t* end, std::uint64_t* values);
  void sort_by_key(std::uint32_t* begin, std::uint32_t* end, std::uint64_t* values);
  void sort_by_key(std::uint64_t* begin, std::uint64_t* end, std::uint64_t* values);
}
