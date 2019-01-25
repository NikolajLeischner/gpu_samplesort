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
    __global__ void scatter(const KeyType *__restrict__ keysInput, int minPos, int maxPos,
                            KeyType *__restrict__ keysOutput,
                            const int *__restrict__ globalBuckets,
                            int *__restrict__ newBucketBounds,
                            int keysPerThread,
                            CompType comp) {
        const int from = blockIdx.x * keysPerThread * FIND_THREADS + minPos;
        const int to = blockIdx.x + 1 == gridDim.x ? maxPos : from + keysPerThread * FIND_THREADS;

        // Read bucket positions. Special treatment for CTA 0. It would read across the start boundary of the buckets array otherwise.
        __shared__ int buckets[K * COUNTERS];
        for (int i = blockIdx.x == 0 ? threadIdx.x + 1 : threadIdx.x; i < K; i += CTA_SIZE) {
            for (int j = 0; j < COUNTERS; ++j)
                buckets[i * COUNTERS + j] = globalBuckets[(i * gridDim.x * COUNTERS) + blockIdx.x * COUNTERS + j - 1];
        }

        // Write the first entries for block 0.
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            buckets[0] = 0;
            for (int j = 1; j < COUNTERS; ++j)
                buckets[j] = globalBuckets[j - 1];
        }

        __shared__ KeyType bst[K];

        KeyType *constBst = reinterpret_cast<KeyType *>(bst_cache);

        if (!DEGENERATED) {
            for (int i = threadIdx.x; i < K; i += CTA_SIZE) bst[i] = constBst[i];
        }
            // All splitters for the bucket are identical, don't even load the bst but just one splitter.
        else if (threadIdx.x == 0) bst[0] = constBst[0];
        __syncthreads();

        // Read keys, get their bucket id and scatter them in batches.
        for (int i = from + threadIdx.x; i < to; i += CTA_SIZE) {
            KeyType d = keysInput[i];
            int bucketPos = 1;

            if (!DEGENERATED) {
                // Traverse bst.
                for (int j = 0; j < LOG_K; ++j) {
                    if (comp(bst[bucketPos - 1], d)) bucketPos = (bucketPos << 1) + 1;
                    else bucketPos <<= 1;
                }
                bucketPos -= K;
            } else {
                if (comp(bst[0], d)) bucketPos = 2;
                else if (comp(d, bst[0])) bucketPos = 0;
            }
            keysOutput[minPos + atomicAdd(buckets + bucketPos * COUNTERS + threadIdx.x % COUNTERS, 1)] = d;
        }

        // The first CTA writes the new bucket boundaries.
        if (blockIdx.x == 0) {
            for (int i = threadIdx.x; i < K; i += CTA_SIZE)
                newBucketBounds[i] =
                        minPos + globalBuckets[(i * gridDim.x * COUNTERS) + (gridDim.x - 1) * COUNTERS + COUNTERS - 1];
        }
    }

    // Same as above but for key-value-pairs.
    template<int K, int LOG_K, int FIND_THREADS, int CTA_SIZE, int COUNTERS, bool DEGENERATED, typename KeyType, typename ValueType, typename CompType>
    __global__ void
    scatter(const KeyType *__restrict__ keysInput, const ValueType *__restrict__ valuesInput,
            int minPos,
            int maxPos, KeyType *__restrict__ keysOutput, ValueType *__restrict__ valuesOutput,
            const int *__restrict__ globalBuckets, int *__restrict__ newBucketBounds,
            int keysPerThread, CompType comp) {
        const int from = blockIdx.x * keysPerThread * FIND_THREADS + minPos;
        const int to = blockIdx.x + 1 == gridDim.x ? maxPos : from + keysPerThread * FIND_THREADS;

        // Read bucket positions. Special treatment for CTA 0. It would read across the start boundary of the buckets array otherwise.
        __shared__ int buckets[K * COUNTERS];
        for (int i = blockIdx.x == 0 ? threadIdx.x + 1 : threadIdx.x; i < K; i += CTA_SIZE) {
            for (int j = 0; j < COUNTERS; ++j)
                buckets[i * COUNTERS + j] = globalBuckets[(i * gridDim.x * COUNTERS) + blockIdx.x * COUNTERS + j - 1];
        }

        // Write the first entries for block 0.
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            buckets[0] = 0;
            for (int j = 1; j < COUNTERS; ++j)
                buckets[j] = globalBuckets[j - 1];
        }

        // Shared memory copy of the search tree.
        __shared__ KeyType bst[K];
        __shared__ KeyType splitter;

        KeyType *constBst = reinterpret_cast<KeyType *>(bst_cache);

        if (!DEGENERATED) {
            for (int i = threadIdx.x; i < K; i += CTA_SIZE) bst[i] = constBst[i];
        }
            // All splitters for the bucket are identical, don't even load the bst but just one splitter.
        else if (threadIdx.x == 0) splitter = bst[0];
        __syncthreads();

        // Read keys, get their bucket id and scatter them in batches.
        for (int i = from + threadIdx.x; i < to; i += CTA_SIZE) {
            KeyType d = keysInput[i];
            ValueType vd = valuesInput[i];
            int bucketPos = 1;

            if (!DEGENERATED) {
                // Traverse bst.
                for (int j = 0; j < LOG_K; ++j) {
                    if (comp(bst[bucketPos - 1], d)) bucketPos = (bucketPos << 1) + 1;
                    else bucketPos <<= 1;
                }

                bucketPos -= K;
            } else {
                if (comp(splitter, d)) bucketPos = 2;
                else if (comp(d, splitter)) bucketPos = 0;
            }

            int outputPos = minPos + atomicAdd(buckets + bucketPos * COUNTERS + threadIdx.x % COUNTERS, 1);
            keysOutput[outputPos] = d;
            valuesOutput[outputPos] = vd;
        }

        // The first CTA writes the new bucket boundaries.
        if (blockIdx.x == 0) {
            for (int i = threadIdx.x; i < K; i += CTA_SIZE)
                newBucketBounds[i] =
                        minPos + globalBuckets[(i * gridDim.x * COUNTERS) + (gridDim.x - 1) * COUNTERS + COUNTERS - 1];
        }
    }
}