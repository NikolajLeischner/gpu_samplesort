#pragma once

namespace SampleSort {
    template<typename T, typename BinaryPredicate>
    __device__ T min(const T &lhs, const T &rhs, BinaryPredicate comp) {
        return comp(lhs, rhs) ? lhs : rhs;
    }

    template<typename T>
    __device__ T min(const T &lhs, const T &rhs) {
        return lhs < rhs ? lhs : rhs;
    }

    template<typename T, typename BinaryPredicate>
    __device__ T max(const T &lhs, const T &rhs, BinaryPredicate comp) {
        return comp(lhs, rhs) ? rhs : lhs;
    }

    template<typename T>
    __device__ T max(const T &lhs, const T &rhs) {
        return lhs < rhs ? rhs : lhs;
    }
}
