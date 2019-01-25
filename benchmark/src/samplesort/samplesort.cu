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

#include <iostream>
#include <algorithm>
#include <stack>
#include <vector>
#include <queue>
#include <random>
#include <limits>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/extrema.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/type_traits.h>
#include <thrust/functional.h>

#include "detail/constants.h"
#include "detail/bucket.h"
#include "detail/create_bst.h"
#include "detail/find_buckets.h"
#include "detail/scatter.h"
#include "detail/quicksort.h"
#include "detail/copy_buckets.h"
#include "detail/temporary_device_memory.h"

namespace SampleSort {

    int clamp(int value, int lo, int hi) {
        return std::max(lo, std::min(value, hi));
    }

    template<int COPY_THREADS, size_t MAX_BLOCK_COUNT, bool KEYS_ONLY, typename KeyType, typename ValueType>
    void move_to_output(std::priority_queue<Bucket> &swappedBuckets, KeyType *dKeys,
            const TemporaryDeviceMemory<KeyType> &keysBuffer, ValueType *dValues,
            const TemporaryDeviceMemory<ValueType> &valuesBuffer) {
        int batchSize = std::min(swappedBuckets.size(), MAX_BLOCK_COUNT);
        TemporaryDeviceMemory<Bucket> devSwappedBucketData((size_t) batchSize);
        std::vector<Bucket> swappedBucketData;

        while (!swappedBuckets.empty()) {
            swappedBucketData.clear();
            batchSize = std::min(swappedBuckets.size(), MAX_BLOCK_COUNT);

            for (int i = 0; i < batchSize; ++i) {
                swappedBucketData.push_back(swappedBuckets.top());
                swappedBuckets.pop();
            }

            devSwappedBucketData.copy_to_device(swappedBucketData.data());

            if (KEYS_ONLY)
                copy_buckets<COPY_THREADS> <<<batchSize, COPY_THREADS>>>
                        (dKeys, keysBuffer.data, devSwappedBucketData.data);
            else
                copy_buckets<COPY_THREADS> <<<batchSize, COPY_THREADS>>>
                        (dKeys, keysBuffer.data, dValues, valuesBuffer.data, devSwappedBucketData.data);
        }
    }

    template<int SORT_THREADS, size_t MAX_BLOCK_COUNT, bool KEYS_ONLY, typename KeyType, typename ValueType, typename CompType>
    void sort_buckets(std::priority_queue<Bucket> &smallBuckets, KeyType *dKeys,
                        const TemporaryDeviceMemory<KeyType> &keysBuffer, ValueType *dValues,
                        const TemporaryDeviceMemory<ValueType> &valuesBuffer, CompType comp) {
        // Below this size odd-even-merge-sort is used in the CTA sort.
        const unsigned int LOCAL_SORT_SIZE = 2048;
        // Might want to choose a different size for key-value sorting, since the
        // shared memory requirements are higher.
        const unsigned int LOCAL_SORT_SIZE_KV = 2048;
        int batchSize = std::min(smallBuckets.size(), MAX_BLOCK_COUNT);
        TemporaryDeviceMemory<Bucket> devSmallBucketData((size_t) batchSize);
        std::vector<Bucket> smallBucketData;

        while (!smallBuckets.empty()) {
            smallBucketData.clear();
            batchSize = std::min(smallBuckets.size(), MAX_BLOCK_COUNT);

            for (int i = 0; i < batchSize; ++i) {
                smallBucketData.push_back(smallBuckets.top());
                smallBuckets.pop();
            }

            devSmallBucketData.copy_to_device(smallBucketData.data());

            if (KEYS_ONLY)
                quicksort<LOCAL_SORT_SIZE, SORT_THREADS> <<<batchSize, SORT_THREADS>>>
                        (dKeys, keysBuffer.data, devSmallBucketData.data, comp);
            else
                quicksort<LOCAL_SORT_SIZE_KV, SORT_THREADS> <<<batchSize, SORT_THREADS>>>
                        (dKeys, keysBuffer.data, dValues, valuesBuffer.data, devSmallBucketData.data, comp);
        }
    }

    template<typename RandomAccessIterator1, typename RandomAccessIterator2, typename StrictWeakOrdering, bool KEYS_ONLY>
    void sort(RandomAccessIterator1 keysBegin, RandomAccessIterator1 keysEnd,
              RandomAccessIterator2 valuesBegin, StrictWeakOrdering comp) {
        const unsigned int A = 32;
        const int LARGE_A = A;
        // Smaller oversampling factor, used when all buckets are smaller than some size.
        const int SMALL_A = A / 2;
        // How large should the largest bucket be to allow using the smaller oversampling factor?
        const int SMALL_A_LIMIT = 1 << 25;
        // Number of replicated bucket counters per thread block in the bucket finding / scattering kernels.
        const int COUNTERS = 8;
        // Factor for additional counter replication in the bucket finding kernel.
        const int COUNTER_COPIES = 1;

        const unsigned int LOCAL_SORT_SIZE = 2048;
        const int BST_THREADS = 128;
        const int FIND_THREADS = 128;
        const int SCATTER_THREADS = 128;
        // Must be a power of 2.
        const int LOCAL_THREADS = 256;
        const int COPY_THREADS = 128;

        // The number of elements/thread is chosen so that at least this many CTAs are used, if possible.
        const int DESIRED_CTA_COUNT = 1024;

        const size_t MAX_BLOCK_COUNT = (1 << 29) - 1;

        typedef typename thrust::iterator_traits<RandomAccessIterator1>::value_type KeyType;
        typedef typename thrust::iterator_traits<RandomAccessIterator2>::value_type ValueType;
        typedef typename StrictWeakOrdering CompType;

        KeyType *dKeys = thrust::raw_pointer_cast(&*keysBegin);
        ValueType *dValues = thrust::raw_pointer_cast(&*valuesBegin);

        const KeyType size = keysEnd - keysBegin;
        if (size == 0) return;

        const int maxBucketSize = clamp((int) size / (2 * std::sqrt((float) K)), 1 << 14, 1 << 18);

        std::stack<Bucket> largeBuckets;
        // Buckets are ordered by size, which improves the performance of the
        // CTA level sorting. Helps the gpu's scheduler?
        std::priority_queue<Bucket> smallBuckets;
        std::priority_queue<Bucket> swappedBuckets;

        // Push the whole input on a stack.
        Bucket init(0, size);

        if (size < (unsigned int) maxBucketSize) smallBuckets.push(init);
        else largeBuckets.push(init);

        TemporaryDeviceMemory<KeyType> keysBuffer(size);

        std::random_device rd;
        std::mt19937 gen(rd());
        gen.seed(17);
        std::uniform_int_distribution<int> distribution;
        Lrand48 *rng = new Lrand48();

        TemporaryDeviceMemory<ValueType> valuesBuffer(KEYS_ONLY ? size : 0);

        // Cooperatively k-way split large buckets. Search tree creation is done for several large buckets in parallel.
        while (!largeBuckets.empty()) {
            // Grab as many large buckets as possible, within the CTA count limitation for a kernel call.
            std::vector<Bucket> buckets;
            int maxNumBlocks = 0;
            while (!largeBuckets.empty() && buckets.size() < MAX_BLOCK_COUNT) {
                Bucket b = largeBuckets.top();

                // Adjust the number of elements/thread according to the bucket size.
                int elementsPerThread = std::max(1, (int) ceil((double) b.size / (DESIRED_CTA_COUNT * FIND_THREADS)));
                int bucketBlocks = (int) ceil(((double) b.size / (elementsPerThread * FIND_THREADS)));

                b.elementsPerThread = elementsPerThread;
                maxNumBlocks = std::max(maxNumBlocks, bucketBlocks);
                buckets.push_back(b);
                largeBuckets.pop();
            }

            // Copy bucket parameters to the GPU.
            TemporaryDeviceMemory<Bucket> devBucketParams(buckets.size());
            devBucketParams.copy_to_device(buckets.data());

            // Create the binary search trees.
            TemporaryDeviceMemory<KeyType> bst(K * buckets.size());

            rng->init((int) buckets.size() * BST_THREADS, distribution(gen));

            // One CTA creates the search tree for one bucket. In the first step only
            // one multiprocessor will be occupied. If no bucket is larger than a certain size,
            // use less oversampling.
            if (maxBucketSize < SMALL_A_LIMIT) {
                TemporaryDeviceMemory<KeyType> sample(SMALL_A * K * buckets.size());
                TemporaryDeviceMemory<KeyType> sampleBuffer(SMALL_A * K * buckets.size());
                create_bst<K, SMALL_A, BST_THREADS, LOCAL_SORT_SIZE> <<<buckets.size(), BST_THREADS>>>
                        (dKeys, keysBuffer.data, devBucketParams.data, bst.data, sample.data, sampleBuffer.data, *rng, comp);
            } else {
                TemporaryDeviceMemory<KeyType> sample(LARGE_A * K * buckets.size());
                TemporaryDeviceMemory<KeyType> sampleBuffer(LARGE_A * K * buckets.size());
                create_bst<K, LARGE_A, BST_THREADS, LOCAL_SORT_SIZE> <<<buckets.size(), BST_THREADS>>>
                        (dKeys, keysBuffer.data, devBucketParams.data, bst.data, sample.data, sampleBuffer.data, *rng, comp);
            }

            rng->destroy();

            // Fetch the bucket parameters again which now contain information about which buckets
            // have only equal splitters. Would be sufficient to just fetch an array of bool flags instead
            // of all parameters. But from profiling it looks as if that would be over-optimization.
            devBucketParams.copy_to_host(buckets.data());

            TemporaryDeviceMemory<int> devBucketCounters((size_t) K * COUNTERS * maxNumBlocks);

            std::vector<int> newBucketBounds(K * buckets.size());
            TemporaryDeviceMemory<int> devNewBucketBounds(K * buckets.size());

            // Loop over the large buckets. The limit for considering a bucket to be large should ensure
            // that the bucket-finding and scattering kernels are launched with a sufficient number of CTAs
            // to make use of all available multiprocessors.
            for (int i = 0; i < buckets.size(); ++i) {
                Bucket b = buckets[i];

                int blockCount = (int) ceil((double) b.size / (FIND_THREADS * b.elementsPerThread));

                int start = b.start;
                int end = b.start + b.size;

                KeyType *input = b.flipped ? keysBuffer.data : dKeys;
                KeyType *output = b.flipped ? dKeys : keysBuffer.data;
                ValueType *valuesInput = b.flipped ? valuesBuffer.data : dValues;
                ValueType *valuesOutput = b.flipped ? dValues : valuesBuffer.data;

                cudaMemcpyToSymbol(bst_cache, bst.data + K * i, K * sizeof(KeyType), 0, cudaMemcpyDeviceToDevice);

                // If all keys in the sample are equal, check if the whole bucket contains only one key.
                if (b.degenerated) {
                    thrust::device_ptr <KeyType> devInput(input + start);
                    KeyType minKey, maxKey;
                    cudaMemcpy(&minKey, thrust::min_element(devInput, devInput + b.size).get(), sizeof(KeyType),
                               cudaMemcpyDeviceToHost);
                    cudaMemcpy(&maxKey, thrust::max_element(devInput, devInput + b.size).get(), sizeof(KeyType),
                               cudaMemcpyDeviceToHost);

                    if (!comp(minKey, maxKey) && !comp(maxKey, minKey)) {
                        buckets[i].constant = true;
                        // Skip the rest, the bucket is already sorted.
                        continue;
                    }
                }

                // Find buckets.
                if (!b.degenerated)
                    find_buckets<K, LOG_K, FIND_THREADS, COUNTERS, COUNTER_COPIES, false>
                            <<<blockCount, FIND_THREADS>>> (input, start, end, devBucketCounters.data, b.elementsPerThread, comp);
                else
                    find_buckets<K, LOG_K, FIND_THREADS, COUNTERS, COUNTER_COPIES, true>
                            <<<blockCount, FIND_THREADS>>> (input, start, end, devBucketCounters.data, b.elementsPerThread, comp);

                // Scan over the bucket counters, yielding the array positions the blocks of the scattering kernel need to write to.
                thrust::device_ptr<int> devCounters(devBucketCounters.data);
                thrust::inclusive_scan(devCounters, devCounters + K * COUNTERS * blockCount, devCounters);

                if (KEYS_ONLY) {
                    if (!b.degenerated)
                        scatter<K, LOG_K, FIND_THREADS, SCATTER_THREADS, COUNTERS, false>
                                <<<blockCount, SCATTER_THREADS>>> (input, start, end, output, devBucketCounters.data,
                                        devNewBucketBounds.data + K * i, b.elementsPerThread, comp);
                    else
                        scatter<K, LOG_K, FIND_THREADS, SCATTER_THREADS, COUNTERS, true>
                                <<<blockCount, SCATTER_THREADS >>> (input, start, end, output, devBucketCounters.data,
                                        devNewBucketBounds.data + K * i, b.elementsPerThread, comp);
                } else {
                    if (!b.degenerated)
                        scatter<K, LOG_K, FIND_THREADS, SCATTER_THREADS, COUNTERS, false>
                                <<<blockCount, SCATTER_THREADS>>> (input, valuesInput, start, end, output, valuesOutput,
                                        devBucketCounters.data, devNewBucketBounds.data + K * i, b.elementsPerThread, comp);
                    else
                        scatter<K, LOG_K, FIND_THREADS, SCATTER_THREADS, COUNTERS, true>
                                <<<blockCount, SCATTER_THREADS>>> (input, valuesInput, start, end, output, valuesOutput,
                                        devBucketCounters.data, devNewBucketBounds.data + K * i, b.elementsPerThread, comp);
                }
            }

            devNewBucketBounds.copy_to_host(newBucketBounds.data());

            for (int i = 0; i < buckets.size(); i++) {
                if (!buckets[i].degenerated) {
                    for (int j = 0; j < K; j++) {
                        int start = (j > 0) ? newBucketBounds[K * i + j - 1] : buckets[i].start;
                        int bucketSize = newBucketBounds[K * i + j] - start;
                        Bucket newBucket(start, bucketSize, !buckets[i].flipped);

                        // Depending on it's size push the bucket on a different stack.
                        if (newBucket.size > maxBucketSize) largeBuckets.push(newBucket);
                        else if (newBucket.size > 1) smallBuckets.push(newBucket);
                        else if (newBucket.size == 1 && newBucket.flipped) swappedBuckets.push(newBucket);
                    }
                } else if (!buckets[i].constant) {
                    // There are only 3 buckets if all splitters were equal.
                    for (int j = 0; j < 3; j++) {
                        int start = (j > 0) ? newBucketBounds[K * i + j - 1] : buckets[i].start;
                        int bucketSize = newBucketBounds[K * i + j] - start;
                        Bucket newBucket(start, bucketSize, !buckets[i].flipped);

                        // Bucket with id 1 contains only equal keys, there is no need to sort it.
                        if (j == 1) {
                            if (newBucket.flipped) swappedBuckets.push(newBucket);
                        } else if (newBucket.size > maxBucketSize) largeBuckets.push(newBucket);
                        else if (newBucket.size > 1) smallBuckets.push(newBucket);
                        else if (newBucket.size == 1 && newBucket.flipped) swappedBuckets.push(newBucket);
                    }
                } else {
                    // The bucket only contains equal keys. No need for sorting.
                    if (buckets[i].flipped) swappedBuckets.push(buckets[i]);
                }
            }
        }
        delete rng;

        move_to_output<COPY_THREADS, MAX_BLOCK_COUNT, KEYS_ONLY>
                (swappedBuckets, dKeys, keysBuffer, dValues, valuesBuffer);

        sort_buckets<LOCAL_THREADS, MAX_BLOCK_COUNT, KEYS_ONLY>
                (smallBuckets, dKeys, keysBuffer, dValues, valuesBuffer, comp);
    }

    void sort_by_key(std::uint16_t *keys, std::uint16_t *keysEnd, std::uint64_t *values) {
        SampleSort::sort<std::uint16_t *, std::uint64_t *, thrust::less<std::uint16_t>, false>
                (keys, keysEnd, values, thrust::less<std::uint16_t>());
    }

    void sort_by_key(std::uint32_t *keys, std::uint32_t *keysEnd, std::uint64_t *values) {
        SampleSort::sort<std::uint32_t *, std::uint64_t *, thrust::less<std::uint32_t>, false>
                (keys, keysEnd, values, thrust::less<std::uint32_t>());
    }

    void sort_by_key(std::uint64_t *keys, std::uint64_t *keysEnd, std::uint64_t *values) {
        SampleSort::sort<std::uint64_t *, std::uint64_t *, thrust::less<std::uint64_t>, false>
                (keys, keysEnd, values, thrust::less<std::uint64_t>());
    }

    void sort(std::uint16_t *keys, std::uint16_t *keysEnd) {
        SampleSort::sort<std::uint16_t *, std::uint16_t *, thrust::less<std::uint16_t>, true>
                (keys, keysEnd, 0, thrust::less<std::uint16_t>());
    }

    void sort(std::uint32_t *keys, std::uint32_t *keysEnd) {
        SampleSort::sort<std::uint32_t *, std::uint32_t *, thrust::less<std::uint32_t>, true>
                (keys, keysEnd, 0, thrust::less<std::uint32_t>());
    }

    void sort(std::uint64_t *keys, std::uint64_t *keysEnd) {
        SampleSort::sort<std::uint64_t *, std::uint64_t *, thrust::less<std::uint64_t>, true>
                (keys, keysEnd, 0, thrust::less<std::uint64_t>());
    }
}