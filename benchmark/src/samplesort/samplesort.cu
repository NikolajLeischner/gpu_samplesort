/**
* GPU Sample Sort
* -----------------------
* Copyright (c) 2009-2010 Nikolaj Leischner and Vitaly Osipov
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

#include "samplesort/detail/Bucket.inl"
#include "samplesort/detail/GlobalCreateBst.inl"
#include "samplesort/detail/GlobalFindBuckets.inl"
#include "samplesort/detail/GlobalScatterElements.inl"
#include "samplesort/detail/CTASampleSort.inl"
#include "samplesort/detail/CTACopyBuckets.inl"

namespace SampleSort {
    bool operator<(const Bucket &lhs, const Bucket &rhs) {
        return rhs.size > lhs.size;
    }

    template<typename RandomAccessIterator1, typename RandomAccessIterator2, typename StrictWeakOrdering,
            unsigned int A, unsigned int LOCAL_SORT_SIZE, unsigned int LOCAL_SORT_SIZE_KV>
    void sort(RandomAccessIterator1 keysBegin, RandomAccessIterator1 keysEnd,
              RandomAccessIterator2 valuesBegin, StrictWeakOrdering comp, bool keysOnly, int numCTAs) {
        const int LARGE_A = A;
        // Smaller oversampling factor, used when all buckets are smaller than some size.
        const int SMALL_A = A / 2;
        // How large should the largest bucket be to allow using the smaller oversampling factor?
        const int SMALL_A_LIMIT = 1 << 25;
        // Number of replicated bucket counters per thread block in the bucket finding / scattering kernels.
        const int COUNTERS = 8;
        // Factor for additional counter replication in the bucket finding kernel.
        const int COUNTER_COPIES = 1;

        const int BST_THREADS = 128;
        const int FIND_THREADS = 128;
        const int SCATTER_THREADS = 128;
        // Must be a power of 2.
        const int LOCAL_THREADS = 256;
        const int COPY_THREADS = 128;

        const int THREAD_MIN_ELEMS = 1;
        // The number of elements/thread is chosen so that at least this many CTAs are used, if possible.
        const int DESIRED_CTA_COUNT = numCTAs;

        const int maxBlockCount = (1 << 16) - 1;

        typedef typename thrust::iterator_traits<RandomAccessIterator1>::value_type KeyType;
        typedef typename thrust::iterator_traits<RandomAccessIterator2>::value_type ValueType;

        KeyType *dKeys = thrust::raw_pointer_cast(&*keysBegin);
        ValueType *dValues = thrust::raw_pointer_cast(&*valuesBegin);

        const KeyType size = keysEnd - keysBegin;
        if (size == 0) return;

        // For small inputs the maximum bucket size is decreased.
        int minMaxBucketSize = 1 << 14;
        // Below this size CTA level quick sort is used.
        int maxBucketSize = 1 << 18;
        // Adjust the maximum bucket size for small inputs.
        maxBucketSize = std::min(maxBucketSize, (int) (size / (2 * std::sqrt((float) K))));
        maxBucketSize = std::max(minMaxBucketSize, maxBucketSize);

        std::stack<Bucket> largeBuckets;
        std::vector<Bucket> curBuckets;
        // Buckets are ordered by size, which improves the performance of the
        // CTA level sorting. Helps the gpu's scheduler?
        std::priority_queue<Bucket> smallBuckets;
        std::priority_queue<Bucket> swappedBuckets;

        // Push the whole input on a stack.
        Bucket init;
        init.start = 0;
        init.size = size;
        init.flipped = false;

        if (size < (unsigned int) maxBucketSize) smallBuckets.push(init);
        else largeBuckets.push(init);

        KeyType *dKeysBuffer = 0;
        cudaMalloc((void **) &dKeysBuffer, size * sizeof(KeyType));

        std::random_device rd;
        std::mt19937 gen(rd());
        gen.seed(17);
        std::uniform_int_distribution<int> distribution;
        Lrand48 *rng = new Lrand48();

        ValueType *dValuesBuffer = 0;
        if (!keysOnly) cudaMalloc((void **) &dValuesBuffer, size * sizeof(ValueType));

        // Cooperatively k-way split large buckets. Search tree creation is done for several large buckets in parallel.
        while (!largeBuckets.empty()) {
            // Grab as many large buckets as possible, within the CTA count limitation for a kernel call.
            curBuckets.clear();
            int maxNumBlocks = 0;
            while (!largeBuckets.empty() && curBuckets.size() < maxBlockCount) {
                Bucket b = largeBuckets.top();

                // Adjust the number of elements/thread according to the bucket size.
                int elementsPerThread = THREAD_MIN_ELEMS;
                elementsPerThread = std::max(elementsPerThread,
                                             (int) ceil((double) b.size / (DESIRED_CTA_COUNT * FIND_THREADS)));
                int bucketBlocks = (int) ceil(((double) b.size / (elementsPerThread * FIND_THREADS)));

                b.elementsPerThread = elementsPerThread;
                maxNumBlocks = std::max(maxNumBlocks, bucketBlocks);
                curBuckets.push_back(b);
                largeBuckets.pop();
            }

            // Copy bucket parameters to the GPU.
            const size_t numCurBuckets = curBuckets.size();
            Bucket *bucketParams = new Bucket[numCurBuckets];
            for (int i = 0; i < (int) curBuckets.size(); ++i)
                bucketParams[i] = curBuckets[i];
            Bucket *dBucketParams = 0;
            cudaMalloc((void **) &dBucketParams, numCurBuckets * sizeof(Bucket));
            cudaMemcpy(dBucketParams, bucketParams, numCurBuckets * sizeof(Bucket), cudaMemcpyHostToDevice);

            // Create the binary search trees.
            KeyType *dBst = 0;
            KeyType *dSample = 0;
            KeyType *dSampleBuffer = 0;
            cudaMalloc((void **) &dBst, K * numCurBuckets * sizeof(KeyType));

            rng->init(numCurBuckets * BST_THREADS, distribution(gen));

            // One CTA creates the search tree for one bucket. In the first step only
            // one multiprocessor will be occupied. If no bucket is larger than a certain size,
            // use less oversampling.
            if (maxBucketSize < SMALL_A_LIMIT) {
                cudaMalloc((void **) &dSample, SMALL_A * K * numCurBuckets * sizeof(KeyType));
                cudaMalloc((void **) &dSampleBuffer, SMALL_A * K * numCurBuckets * sizeof(KeyType));
                globalCreateBst<KeyType, StrictWeakOrdering, K, SMALL_A, BST_THREADS, LOCAL_SORT_SIZE> << <
                numCurBuckets,
                        BST_THREADS >> > (dKeys, dKeysBuffer, dBucketParams, dBst, dSample, dSampleBuffer, *rng, comp);
            } else {
                cudaMalloc((void **) &dSample, LARGE_A * K * numCurBuckets * sizeof(KeyType));
                cudaMalloc((void **) &dSampleBuffer, LARGE_A * K * numCurBuckets * sizeof(KeyType));
                globalCreateBst<KeyType, StrictWeakOrdering, K, LARGE_A, BST_THREADS, LOCAL_SORT_SIZE> << <
                numCurBuckets,
                        BST_THREADS >> > (dKeys, dKeysBuffer, dBucketParams, dBst, dSample, dSampleBuffer, *rng, comp);
            }

            rng->destroy();
            cudaFree(dSample);
            cudaFree(dSampleBuffer);

            // Fetch the bucket parameters again which now contain information about which buckets
            // have only equal splitters. Would be sufficient to just fetch an array of bool flags instead
            // of all parameters. But from profiling it looks as if that would be over-optimization.
            cudaMemcpy(bucketParams, dBucketParams, numCurBuckets * sizeof(Bucket), cudaMemcpyDeviceToHost);

            int *dBucketCounters = 0;
            cudaMalloc((void **) &dBucketCounters, K * COUNTERS * maxNumBlocks * sizeof(int));

            int *newBucketBounds = new int[K * numCurBuckets];
            int *dNewBucketBounds = 0;
            cudaMalloc((void **) &dNewBucketBounds, K * numCurBuckets * sizeof(int));

            // Loop over the large buckets. The limit for considering a bucket to be large should ensure
            // that the bucket-finding and scattering kernels are launched with a sufficient number of CTAs
            // to make use of all available multiprocessors.
            for (int i = 0; i < numCurBuckets; ++i) {
                Bucket b = bucketParams[i];
                KeyType *input;
                KeyType *output;
                ValueType *valuesInput;
                ValueType *valuesOutput;

                int numCurBucketCTAs = (int) ceil((double) b.size / (FIND_THREADS * b.elementsPerThread));

                if (b.flipped) {
                    input = dKeysBuffer;
                    output = dKeys;
                    valuesInput = dValuesBuffer;
                    valuesOutput = dValues;
                }
                else {
                    input = dKeys;
                    output = dKeysBuffer;
                    valuesInput = dValues;
                    valuesOutput = dValuesBuffer;
                }

                cudaMemcpyToSymbol(bstCache, dBst + K * i, K * sizeof(KeyType), 0, cudaMemcpyDeviceToDevice);

                // If all keys in the sample are equal, we might be persuaded to assume that all keys in the bucket are equal.
                // Two reductions tell us if this is the case.
                if (b.degenerated) {
                    thrust::device_ptr <KeyType> devInput(input + b.start);
                    KeyType minKey, maxKey;
                    cudaMemcpy(&minKey, thrust::min_element(devInput, devInput + b.size).get(), sizeof(KeyType),
                               cudaMemcpyDeviceToHost);
                    cudaMemcpy(&maxKey, thrust::max_element(devInput, devInput + b.size).get(), sizeof(KeyType),
                               cudaMemcpyDeviceToHost);

                    if (!comp(minKey, maxKey) && !comp(maxKey, minKey)) {
                        bucketParams[i].constant = true;
                        // Skip the bucket finding and scattering.
                        continue;
                    }
                }

                // Find buckets.
                if (!b.degenerated)
                    globalFindBuckets<KeyType, StrictWeakOrdering, K, LOG_K, FIND_THREADS, COUNTERS, COUNTER_COPIES, false>
                            << < numCurBucketCTAs, FIND_THREADS >> >
                                                   (input, b.start, b.start +
                                                                    b.size, dBucketCounters, b.elementsPerThread, comp);
                else
                    globalFindBuckets<KeyType, StrictWeakOrdering, K, LOG_K, FIND_THREADS, COUNTERS, COUNTER_COPIES, true>
                            << < numCurBucketCTAs, FIND_THREADS >> >
                                                   (input, b.start, b.start +
                                                                    b.size, dBucketCounters, b.elementsPerThread, comp);

                // Scan over the bucket counters, yielding the array positions the blocks of the scattering kernel need to write to.
                thrust::device_ptr<int> devCounters(dBucketCounters);
                thrust::inclusive_scan(devCounters, devCounters + K * COUNTERS * numCurBucketCTAs, devCounters);

                // Scatter elements, extract bucket positions.
                if (keysOnly) {
                    if (!b.degenerated)
                        globalScatterElements<KeyType, StrictWeakOrdering, K, LOG_K, FIND_THREADS, SCATTER_THREADS, COUNTERS, false>
                                << < numCurBucketCTAs,
                                SCATTER_THREADS >> >
                                (input, b.start, b.start + b.size, output, dBucketCounters, dNewBucketBounds + K *
                                                                                                               i, b.elementsPerThread, comp);
                    else
                        globalScatterElements<KeyType, StrictWeakOrdering, K, LOG_K, FIND_THREADS, SCATTER_THREADS, COUNTERS, true>
                                << < numCurBucketCTAs,
                                SCATTER_THREADS >> >
                                (input, b.start, b.start + b.size, output, dBucketCounters, dNewBucketBounds + K *
                                                                                                               i, b.elementsPerThread, comp);
                } else {
                    if (!b.degenerated)
                        globalScatterElementsKeyValue<KeyType, ValueType, StrictWeakOrdering, K, LOG_K, FIND_THREADS, SCATTER_THREADS, COUNTERS, false>
                                << < numCurBucketCTAs, SCATTER_THREADS >> > (input, valuesInput, b.start, b.start +
                                                                                                          b.size, output, valuesOutput,
                                dBucketCounters, dNewBucketBounds + K * i, b.elementsPerThread, comp);
                    else
                        globalScatterElementsKeyValue<KeyType, ValueType, StrictWeakOrdering, K, LOG_K, FIND_THREADS, SCATTER_THREADS, COUNTERS, true>
                                << < numCurBucketCTAs, SCATTER_THREADS >> > (input, valuesInput, b.start, b.start +
                                                                                                          b.size, output, valuesOutput,
                                dBucketCounters, dNewBucketBounds + K * i, b.elementsPerThread, comp);
                }
            }

            // Copy bucket positions and parameters to CPU, refill stack.
            cudaMemcpy(newBucketBounds, dNewBucketBounds, K * numCurBuckets * sizeof(int), cudaMemcpyDeviceToHost);

            for (int i = 0; i < numCurBuckets; i++) {
                if (!bucketParams[i].degenerated) {
                    for (int j = 0; j < K; j++) {
                        Bucket newBucket;

                        newBucket.start = (j > 0) ? newBucketBounds[K * i + j - 1] : bucketParams[i].start;
                        newBucket.size = newBucketBounds[K * i + j] - newBucket.start;
                        newBucket.flipped = !bucketParams[i].flipped;

                        // Depending on it's size push the bucket on a different stack.
                        if (newBucket.size > maxBucketSize) largeBuckets.push(newBucket);
                        else if (newBucket.size > 1) smallBuckets.push(newBucket);
                        else if (newBucket.size == 1 && newBucket.flipped) swappedBuckets.push(newBucket);
                    }
                } else if (!bucketParams[i].constant) {
                    // There are only 3 buckets if all splitters were equal.
                    for (int j = 0; j < 3; j++) {
                        Bucket newBucket;

                        newBucket.start = (j > 0) ? newBucketBounds[K * i + j - 1] : bucketParams[i].start;
                        newBucket.size = newBucketBounds[K * i + j] - newBucket.start;
                        newBucket.flipped = !bucketParams[i].flipped;

                        // Bucket with id 1 contains only equal keys, there is no need to sort it.
                        if (j == 1) {
                            if (newBucket.flipped) swappedBuckets.push(newBucket);
                        } else if (newBucket.size > maxBucketSize) largeBuckets.push(newBucket);
                        else if (newBucket.size > 1) smallBuckets.push(newBucket);
                        else if (newBucket.size == 1 && newBucket.flipped) swappedBuckets.push(newBucket);
                    }
                } else {
                    // The bucket only contains equal keys. No need for sorting.
                    if (bucketParams[i].flipped) swappedBuckets.push(bucketParams[i]);
                }
            }

            // Clean up temporary data.
            cudaFree(dNewBucketBounds);
            cudaFree(dBucketCounters);
            cudaFree(dBucketParams);
            cudaFree(dBst);
            delete[] newBucketBounds;
            delete[] bucketParams;
        }

        // Move the constant buckets from the buffer back to the input array. Do it in batches,
        // if their number exceeds the CTA count limit for a kernel call.
        int numSwappedBuckets = swappedBuckets.size();
        Bucket *swappedBucketData = new Bucket[std::min(numSwappedBuckets, maxBlockCount)];
        Bucket *dSwappedBucketData = 0;
        cudaMalloc((void **) &dSwappedBucketData, std::min(numSwappedBuckets, maxBlockCount) * sizeof(Bucket));

        while (numSwappedBuckets > 0) {
            int batchSize = std::min(numSwappedBuckets, maxBlockCount);
            numSwappedBuckets -= batchSize;

            for (int i = 0; i < batchSize; ++i) {
                swappedBucketData[i] = swappedBuckets.top();
                swappedBuckets.pop();
            }

            cudaMemcpy(dSwappedBucketData, swappedBucketData, batchSize * sizeof(Bucket), cudaMemcpyHostToDevice);

            if (keysOnly)
                CTACopyBuckets<COPY_THREADS> << < batchSize, COPY_THREADS >> > (dKeys, dKeysBuffer, dSwappedBucketData);
            else
                CTACopyBucketsKeyValue<COPY_THREADS> << < batchSize, COPY_THREADS >> >
                                                                     (dKeys, dKeysBuffer, dValues, dValuesBuffer, dSwappedBucketData);
        }

        // Now sort the small buckets on the gpu. Again, do it in batches if necessary.
        int numSmallBuckets = smallBuckets.size();
        Bucket *smallBucketData = new Bucket[std::min(numSmallBuckets, maxBlockCount)];
        Bucket *dSmallBucketData = 0;
        cudaMalloc((void **) &dSmallBucketData, std::min(numSmallBuckets, maxBlockCount) * sizeof(Bucket));

        while (numSmallBuckets > 0) {
            int batchSize = std::min(numSmallBuckets, maxBlockCount);
            numSmallBuckets -= batchSize;

            for (int i = 0; i < batchSize; ++i) {
                smallBucketData[i] = smallBuckets.top();
                smallBuckets.pop();
            }

            cudaMemcpy(dSmallBucketData, smallBucketData, batchSize * sizeof(Bucket), cudaMemcpyHostToDevice);

            rng->init(LOCAL_THREADS, distribution(gen));

            if (keysOnly)
                CTASampleSort<KeyType, StrictWeakOrdering, LOCAL_SORT_SIZE, LOCAL_THREADS> << < batchSize,
                        LOCAL_THREADS >> > (dKeys,
                                dKeysBuffer, dSmallBucketData, *rng, comp);
            else
                CTASampleSortKeyValue<KeyType, ValueType, StrictWeakOrdering, LOCAL_SORT_SIZE, LOCAL_THREADS> << <
                batchSize, LOCAL_THREADS >> > (dKeys,
                        dKeysBuffer, dValues, dValuesBuffer, dSmallBucketData, *rng, comp);

            rng->destroy();
        }

        // Clean up.
        delete rng;
        delete[] swappedBucketData;
        delete[] smallBucketData;
        cudaFree(dSwappedBucketData);
        cudaFree(dSmallBucketData);
        cudaFree(dKeysBuffer);

        if (!keysOnly) cudaFree(dValuesBuffer);
    }

    template<typename KeyType, typename ValueType>
    void sortTempl(KeyType *keys, KeyType *keysEnd, ValueType *values = 0) {
        const unsigned int A = 32;
        // Below this size odd-even-merge-sort is used in the CTA sort.
        const unsigned int LOCAL_SORT_SIZE = 2048;
        // Might want to choose a different size for key-value sorting, since the
        // shared memory requirements are higher
        const unsigned int LOCAL_SORT_SIZE_KV = 1784;

        thrust::less <KeyType> comp;

        SampleSort::sort<KeyType *, ValueType *, thrust::less < KeyType>,
                A, LOCAL_SORT_SIZE, LOCAL_SORT_SIZE_KV > (keys, keysEnd, values, comp, values == 0, 400);
    }

    void sort_by_key(std::uint16_t *keys, std::uint16_t *keysEnd, std::uint64_t *values) {
        sortTempl<std::uint16_t, std::uint64_t>(keys, keysEnd, values);
    }

    void sort_by_key(std::uint32_t *keys, std::uint32_t *keysEnd, std::uint64_t *values) {
        sortTempl<std::uint32_t, std::uint64_t>(keys, keysEnd, values);
    }

    void sort_by_key(std::uint64_t *keys, std::uint64_t *keysEnd, std::uint64_t *values) {
        sortTempl<std::uint64_t, std::uint64_t>(keys, keysEnd, values);
    }

    void sort(std::uint16_t *keys, std::uint16_t *keysEnd) {
        sortTempl<std::uint16_t, std::uint16_t>(keys, keysEnd, 0);
    }

    void sort(std::uint32_t *keys, std::uint32_t *keysEnd) {
        sortTempl<std::uint32_t, std::uint32_t>(keys, keysEnd, 0);
    }

    void sort(std::uint64_t *keys, std::uint64_t *keysEnd) {
        sortTempl<std::uint64_t, std::uint64_t>(keys, keysEnd, 0);
    }
}



