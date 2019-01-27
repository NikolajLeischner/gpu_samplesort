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
    // Create bucket counters which are relative to the CTA. To obtain global
    // offsets for scattering a prefix sum has to be performed afterwards.
    // Bucket-finding & scattering must use the same number of elements per thread.
    template<int K, int LOG_K, int CTA_SIZE, int COUNTERS, int COUNTER_COPIES, bool DEGENERATED, typename KeyType, typename CompType>
    __global__ static void find_buckets(
            KeyType *keys,
            int min_pos,
            int max_pos,
            int *global_buckets,
            int keys_per_thread,
            CompType comp) {
        const int LOCAL_COUNTERS = COUNTERS * COUNTER_COPIES;
        // This reduces register usage.
        __shared__ int block;
        __shared__ int grid;
        if (threadIdx.x == 0) {
            block = blockIdx.x;
            grid = gridDim.x;
        }
        __syncthreads();

        const int block_element_count = keys_per_thread * CTA_SIZE;
        const int from = block * block_element_count + min_pos;
        int to = (block + 1 == grid) ? max_pos : from + block_element_count;

        __shared__ KeyType bst[K];
        __shared__ int buckets[K * LOCAL_COUNTERS];

        KeyType *const_bst = reinterpret_cast<KeyType *>(bst_cache);

        for (int i = threadIdx.x; i < K * LOCAL_COUNTERS; i += CTA_SIZE) buckets[i] = 0;

        if (!DEGENERATED) {
            for (int i = threadIdx.x; i < K; i += CTA_SIZE) bst[i] = const_bst[i];
        }
            // All splitters for the bucket are identical, don't even load the bst but just one splitter.
        else if (threadIdx.x == 0) bst[0] = const_bst[0];
        __syncthreads();

        for (int i = from + threadIdx.x; i < to; i += CTA_SIZE) {
            int bucket_pos = 1;
            KeyType d = keys[i];

            if (!DEGENERATED) {
                // Traverse bst.
                for (int j = 0; j < LOG_K; ++j) {
                    if (comp(bst[bucket_pos - 1], d)) bucket_pos = (bucket_pos << 1) + 1;
                    else bucket_pos <<= 1;
                }

                bucket_pos = bucket_pos - K;
            } else {
                if (comp(bst[0], d)) bucket_pos = 2;
                else if (comp(d, bst[0])) bucket_pos = 0;
            }

            atomicAdd(buckets + bucket_pos * LOCAL_COUNTERS + threadIdx.x % LOCAL_COUNTERS, 1);
        }

        __syncthreads();

        // Sum up and write back CTA bucket counters.
        for (int i = threadIdx.x; i < K; i += CTA_SIZE) {
            for (int j = 0; j < COUNTERS; ++j) {
                int b = 0;
                for (int k = 0; k < COUNTER_COPIES; ++k)
                    b += buckets[i * LOCAL_COUNTERS + j + k * COUNTERS];

                global_buckets[(i * grid * COUNTERS) + (block * COUNTERS) + j] = b;
            }
        }
    }
}