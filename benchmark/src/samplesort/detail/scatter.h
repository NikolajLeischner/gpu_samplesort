/**
* GPU Sample Sort
* -----------------------
* Copyright (c) 2009-2019 Nikolaj Leischner and Vitaly Osipov
*
* Permission is hereby granted, free of charge, to any person
* obtaining a copy of this software and associated documentation
* files (the "Software"), to deal in the Software without
* restriction, including without limitation the rights to use,
* copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following
* conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
* OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
* NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
* HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
* WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
* OTHER DEALINGS IN THE SOFTWARE.
**/

#include "constants.h"

namespace SampleSort {

    // Scatter elements, using the previously generated global bucket offsets. Each CTA corresponds
    // to one CTA in the bucket finding kernel, although the number of threads per CTA may differ.
    // Note that the bucket ids have to be generated again. This is faster than storing them in global
    // memory and reading them here again, and avoids memory overhead.
    // Bucket-finding & scattering must use the same number of elements per thread.
    template<int K, int LOG_K, int FIND_THREADS, int CTA_SIZE, int COUNTERS, bool DEGENERATED, typename KeyType, typename CompType>
    __global__ void scatter(
            const KeyType *__restrict__ keys,
            int min_pos,
            int max_pos,
            KeyType *__restrict__ keys_out,
            const int *__restrict__ global_buckets,
            int *__restrict__ bucket_bounds,
            int keys_per_thread,
            CompType comp) {
        const int from = blockIdx.x * keys_per_thread * FIND_THREADS + min_pos;
        const int to = blockIdx.x + 1 == gridDim.x ? max_pos : from + keys_per_thread * FIND_THREADS;

        // Read bucket positions. Special treatment for CTA 0. It would read across the start boundary of the buckets array otherwise.
        __shared__ int buckets[K * COUNTERS];
        for (int i = blockIdx.x == 0 ? threadIdx.x + 1 : threadIdx.x; i < K; i += CTA_SIZE) {
            for (int j = 0; j < COUNTERS; ++j)
                buckets[i * COUNTERS + j] = global_buckets[(i * gridDim.x * COUNTERS) + blockIdx.x * COUNTERS + j - 1];
        }

        // Write the first entries for block 0.
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            buckets[0] = 0;
            for (int j = 1; j < COUNTERS; ++j)
                buckets[j] = global_buckets[j - 1];
        }

        __shared__ KeyType bst[K];

        KeyType *const_bst = reinterpret_cast<KeyType *>(bst_cache);

        if (!DEGENERATED) {
            for (int i = threadIdx.x; i < K; i += CTA_SIZE) bst[i] = const_bst[i];
        }
            // All splitters for the bucket are identical, don't even load the bst but just one splitter.
        else if (threadIdx.x == 0) bst[0] = const_bst[0];
        __syncthreads();

        // Read keys, get their bucket id and scatter them in batches.
        for (int i = from + threadIdx.x; i < to; i += CTA_SIZE) {
            KeyType d = keys[i];
            int position = 1;

            if (!DEGENERATED) {
                // Traverse bst.
                for (int j = 0; j < LOG_K; ++j) {
                    if (comp(bst[position - 1], d)) position = (position << 1) + 1;
                    else position <<= 1;
                }
                position -= K;
            } else {
                if (comp(bst[0], d)) position = 2;
                else if (comp(d, bst[0])) position = 0;
            }
            keys_out[min_pos + atomicAdd(buckets + position * COUNTERS + threadIdx.x % COUNTERS, 1)] = d;
        }

        // The first CTA writes the new bucket boundaries.
        if (blockIdx.x == 0) {
            for (int i = threadIdx.x; i < K; i += CTA_SIZE)
                bucket_bounds[i] =
                        min_pos +
                        global_buckets[(i * gridDim.x * COUNTERS) + (gridDim.x - 1) * COUNTERS + COUNTERS - 1];
        }
    }

    // Same as above but for key-value-pairs.
    template<int K, int LOG_K, int FIND_THREADS, int CTA_SIZE, int COUNTERS, bool DEGENERATED, typename KeyType, typename ValueType, typename CompType>
    __global__ void scatter(
            const KeyType *__restrict__ keys,
            const ValueType *__restrict__ values,
            int min_pos,
            int max_pos,
            KeyType *__restrict__ keys_out,
            ValueType *__restrict__ values_out,
            const int *__restrict__ global_buckets,
            int *__restrict__ bucket_bounds,
            int keys_per_thread,
            CompType comp) {
        const int from = blockIdx.x * keys_per_thread * FIND_THREADS + min_pos;
        const int to = blockIdx.x + 1 == gridDim.x ? max_pos : from + keys_per_thread * FIND_THREADS;

        // Read bucket positions. Special treatment for CTA 0. It would read across the start boundary of the buckets array otherwise.
        __shared__ int buckets[K * COUNTERS];
        for (int i = blockIdx.x == 0 ? threadIdx.x + 1 : threadIdx.x; i < K; i += CTA_SIZE) {
            for (int j = 0; j < COUNTERS; ++j)
                buckets[i * COUNTERS + j] = global_buckets[(i * gridDim.x * COUNTERS) + blockIdx.x * COUNTERS + j - 1];
        }

        // Write the first entries for block 0.
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            buckets[0] = 0;
            for (int j = 1; j < COUNTERS; ++j)
                buckets[j] = global_buckets[j - 1];
        }

        // Shared memory copy of the search tree.
        __shared__ KeyType bst[K];
        __shared__ KeyType splitter;

        KeyType *const_bst = reinterpret_cast<KeyType *>(bst_cache);

        if (!DEGENERATED) {
            for (int i = threadIdx.x; i < K; i += CTA_SIZE) bst[i] = const_bst[i];
        }
            // All splitters for the bucket are identical, don't even load the bst but just one splitter.
        else if (threadIdx.x == 0) splitter = bst[0];
        __syncthreads();

        // Read keys, get their bucket id and scatter them in batches.
        for (int i = from + threadIdx.x; i < to; i += CTA_SIZE) {
            KeyType d = keys[i];
            ValueType vd = values[i];
            int position = 1;

            if (!DEGENERATED) {
                // Traverse bst.
                for (int j = 0; j < LOG_K; ++j) {
                    if (comp(bst[position - 1], d)) position = (position << 1) + 1;
                    else position <<= 1;
                }

                position -= K;
            } else {
                if (comp(splitter, d)) position = 2;
                else if (comp(d, splitter)) position = 0;
            }

            int out_position = min_pos + atomicAdd(buckets + position * COUNTERS + threadIdx.x % COUNTERS, 1);
            keys_out[out_position] = d;
            values_out[out_position] = vd;
        }

        // The first CTA writes the new bucket boundaries.
        if (blockIdx.x == 0) {
            for (int i = threadIdx.x; i < K; i += CTA_SIZE)
                bucket_bounds[i] =
                        min_pos +
                        global_buckets[(i * gridDim.x * COUNTERS) + (gridDim.x - 1) * COUNTERS + COUNTERS - 1];
        }
    }
}