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

#include "bucket.h"
#include "scan.h"
#include "odd_even_sort.h"
#include "predicates.h"

namespace SampleSort {
    // Each CTA performs quicksort on one bucket. Batcher's odd-even-merge-sort is used for sorting
    // small sequences. Based on the GPU quicksort implementation by Daniel Cederman and Philippas Tsigas.
    // Like the original implementation the code tries to obey the memory coalescing rules for the G80
    // architecture. In this case doing so also improves performance on the G200 architecture.
    template<typename KeyType, typename CompType, int LOCAL_SORT_SIZE, unsigned int CTA_SIZE>
    __global__ static void quicksort(KeyType *keysInput, KeyType *keysBuffer, const struct Bucket *bucketParams,
                                     CompType comp) {
        const int sharedSize = LOCAL_SORT_SIZE < 2 * CTA_SIZE ? 2 * CTA_SIZE : LOCAL_SORT_SIZE;
        // Used for shared memory sorting and for holding 3 prefix sum arrays of block size.
        __shared__
        KeyType shared[sharedSize];

        // A stack is used to handle recursion.
        __shared__ int stack;
        __shared__ unsigned int start[32];
        __shared__ unsigned int end[32];
        __shared__ bool flip[32];

        // The total number of small and large elements.
        __shared__ unsigned int smallOffset;
        __shared__ unsigned int largeOffset;

        // The current pivot.
        __shared__ KeyType pivot;

        // The current sequence to be split.
        __shared__ int from;
        __shared__ int to;
        __shared__ int size;

        __shared__ KeyType *keys;
        __shared__ KeyType *keys2;

        // Initialize by putting the whole input on the stack.
        if (threadIdx.x == 0) {
            start[0] = bucketParams[blockIdx.x].start;
            end[0] = bucketParams[blockIdx.x].start + bucketParams[blockIdx.x].size;
            flip[0] = bucketParams[blockIdx.x].flipped;

            stack = 0;
        }

        __syncthreads();

        while (stack >= 0) {
            // Get the next sequence.
            if (threadIdx.x == 0) {
                from = start[stack];
                to = end[stack];
                size = to - from;

                keys = flip[stack] ? keysBuffer : keysInput;
                keys2 = flip[stack] ? keysInput : keysBuffer;
            }

            __syncthreads();

            // If the sequence is small enough it is sorted in shared memory.
            if (size <= LOCAL_SORT_SIZE) {
                if (size > 0) {
                    unsigned int coal = (from) & 0xf;
                    for (int i = threadIdx.x - coal; i < size; i += CTA_SIZE)
                        if (i >= 0) shared[i] = keys[i + from];

                    __syncthreads();

                    // Depending on the size of the sequence to sort, choose a fitting permutation of odd-even-merge-sort.
                    // This case-distinction saves a few expensive instructions at run time. For sort sizes > 2048 more case
                    // distinctions would have to be added. Also a minimum block size of 256 is assumed here.
                    if (size > 1024) {
                        if (CTA_SIZE > 1024) odd_even_sort<KeyType, CompType, 2048>(shared, size, comp);
                        else odd_even_sort<KeyType, CompType, 2048, CTA_SIZE>(shared, size, comp);
                    } else if (size > 512) {
                        if (CTA_SIZE > 512) odd_even_sort<KeyType, CompType, 1024>(shared, size, comp);
                        else odd_even_sort<KeyType, CompType, 1024, CTA_SIZE>(shared, size, comp);
                    } else if (size > 256) {
                        if (CTA_SIZE > 256) odd_even_sort<KeyType, CompType, 512>(shared, size, comp);
                        else odd_even_sort<KeyType, CompType, 512, CTA_SIZE>(shared, size, comp);
                    } else if (size > 128) {
                        if (CTA_SIZE > 128) odd_even_sort<KeyType, CompType, 256>(shared, size, comp);
                        else odd_even_sort<KeyType, CompType, 256, CTA_SIZE>(shared, size, comp);
                    } else if (size > 64) {
                        if (CTA_SIZE > 64) odd_even_sort<KeyType, CompType, 128>(shared, size, comp);
                        else odd_even_sort<KeyType, CompType, 128, CTA_SIZE>(shared, size, comp);
                    } else if (size > 32) {
                        if (CTA_SIZE > 32) odd_even_sort<KeyType, CompType, 64>(shared, size, comp);
                        else odd_even_sort<KeyType, CompType, 64, CTA_SIZE>(shared, size, comp);
                    } else odd_even_sort<KeyType, CompType, 32>(shared, size, comp);

                    for (int i = threadIdx.x; i < size; i += CTA_SIZE)
                        keysInput[i + from] = shared[i];
                }

                if (threadIdx.x == 0)
                    --stack;

                __syncthreads();
            } else {
                if (threadIdx.x == 0) {
                    KeyType mip = SampleSort::min<KeyType>
                                  (SampleSort::min<KeyType> (keys[from], keys[to - 1], comp), keys[(from + to) /
                                                                                                      2], comp);
                    KeyType map = SampleSort::max<KeyType>
                                  (SampleSort::max<KeyType> (keys[from], keys[to - 1], comp), keys[(from + to) /
                                                                                                      2], comp);
                    pivot = SampleSort::min<KeyType>(SampleSort::max<KeyType>(mip / 2 + map / 2, mip, comp), map, comp);
                }

                unsigned int ll = 0;
                unsigned int lr = 0;

                unsigned int coal = (from) & 0xf;

                __syncthreads();

                if (threadIdx.x + from - coal < to) {
                    KeyType d = keys[threadIdx.x + from - coal];

                    if (threadIdx.x >= coal) {
                        if (comp(d, pivot)) ++ll;
                        else if (comp(pivot, d)) ++lr;
                    }
                }

                for (int i = from + threadIdx.x + CTA_SIZE - coal; i < to; i += CTA_SIZE) {
                    KeyType d = keys[i];

                    if (comp(d, pivot)) ++ll;
                    else if (comp(pivot, d)) ++lr;
                }

                // Generate offsets for writing small/large elements for each thread.
                unsigned int *buffer = (unsigned int *) shared;
                buffer[threadIdx.x] = ll;
                __syncthreads();
                brent_kung_inclusive<unsigned int, CTA_SIZE>(buffer);

                unsigned int x = from + buffer[threadIdx.x + 1] - 1;
                __syncthreads();
                if (threadIdx.x == 0) smallOffset = buffer[CTA_SIZE];
                buffer[threadIdx.x] = lr;
                __syncthreads();
                brent_kung_inclusive<unsigned int, CTA_SIZE>(buffer);

                unsigned int y = to - buffer[threadIdx.x + 1];

                if (threadIdx.x == 0) {
                    largeOffset = buffer[CTA_SIZE];

                    // Refill the stack.
                    flip[stack + 1] = !flip[stack];
                    flip[stack] = !flip[stack];

                    if (smallOffset < largeOffset) {
                        start[stack + 1] = start[stack];
                        start[stack] = to - largeOffset;
                        end[stack + 1] = from + smallOffset;
                    } else {
                        end[stack + 1] = end[stack];
                        end[stack] = from + smallOffset;
                        start[stack + 1] = to - largeOffset;
                    }

                    ++stack;
                }

                __syncthreads();


                // Use the offsets generated by the prefix sums to write the elements to their new positions.
                if (threadIdx.x + from - coal < to) {
                    KeyType d = keys[threadIdx.x + from - coal];

                    if (threadIdx.x >= coal) {
                        if (comp(d, pivot)) keys2[x--] = d;
                        else if (comp(pivot, d)) keys2[y++] = d;
                    }
                }

                for (int i = from + threadIdx.x + CTA_SIZE - coal; i < to; i += CTA_SIZE) {
                    KeyType d = keys[i];

                    if (comp(d, pivot)) keys2[x--] = d;
                    else if (comp(pivot, d)) keys2[y++] = d;
                }

                __syncthreads();

                // Write the keys equal to the pivot to the output array since
                // their final position is known.
                for (int i = from + smallOffset + threadIdx.x; i < to - largeOffset; i += CTA_SIZE)
                    keysInput[i] = pivot;
                __syncthreads();
            }
        }
    }

    // Same as above but for key-value-pairs.
    template<typename KeyType, typename ValueType, typename CompType, int LOCAL_SORT_SIZE, unsigned int CTA_SIZE>
    __global__ static void quicksort(KeyType *keysInput, KeyType *keysBuffer, ValueType *valuesInput,
                                     ValueType *valuesBuffer, struct Bucket *bucketParams,
                                     CompType comp) {
        // Used for shared memory sorting and for holding 3 prefix sum arrays of block size.
        const int sharedSize = LOCAL_SORT_SIZE < 2 * (CTA_SIZE + 1) ? 2 * (CTA_SIZE + 1) : LOCAL_SORT_SIZE;
        __shared__ KeyType shared[sharedSize];
        __shared__ ValueType sharedValues[LOCAL_SORT_SIZE];

        // A stack is used to handle recursion.
        __shared__ int stack;
        __shared__ unsigned int start[32];
        __shared__ unsigned int end[32];
        __shared__ bool flip[32];

        // Arrays for calculating prefix sums over the number of smaller/equal/larger elements
        // for each thread.
        unsigned int *smallBlock = (unsigned int *) shared;
        unsigned int *equalBlock = (unsigned int *) (&smallBlock[(CTA_SIZE + 1)]);
        unsigned int *largeBlock = (unsigned int *) sharedValues;

        // The current pivot.
        __shared__ KeyType pivot;

        // The current sequence to be split.
        __shared__ int from;
        __shared__ int to;
        __shared__ int size;

        __shared__ KeyType *keys;
        __shared__ KeyType *keys2;
        __shared__ ValueType *values;
        __shared__ ValueType *values2;

        // Initialize by putting the whole input on the stack.
        if (threadIdx.x == 0) {
            start[0] = bucketParams[blockIdx.x].start;
            end[0] = bucketParams[blockIdx.x].start + bucketParams[blockIdx.x].size;
            flip[0] = bucketParams[blockIdx.x].flipped;

            stack = 0;
        }

        __syncthreads();

        while (stack >= 0) {
            // Get the next sequence.
            if (threadIdx.x == 0) {
                from = start[stack];
                to = end[stack];
                size = to - from;

                if (!flip[stack]) {
                    keys = keysInput;
                    keys2 = keysBuffer;
                    values = valuesInput;
                    values2 = valuesBuffer;
                } else {
                    keys = keysBuffer;
                    keys2 = keysInput;
                    values = valuesBuffer;
                    values2 = valuesInput;
                }
            }

            __syncthreads();

            // If the sequence is small enough it is sorted in shared memory.
            if (size <= LOCAL_SORT_SIZE) {
                if (size > 0) {
                    unsigned int coal = (from) & 0xf;
                    for (int i = threadIdx.x; i < size + coal; i += CTA_SIZE) {
                        KeyType key = keys[i + from - coal];
                        ValueType value = values[i + from - coal];
                        if (i >= coal) {
                            shared[i - coal] = key;
                            sharedValues[i - coal] = value;
                        }
                    }

                    __syncthreads();

                    // Depending on the size of the sequence to sort, choose a fitting permutation of odd-even-merge-sort.
                    // This case-distinction saves a few expensive instructions at run time. For sort sizes > 2048 more case
                    // distinctions would have to be added. Also a minimum block size of 256 is assumed here.
                    if (size > 1024) {
                        if (CTA_SIZE > 1024)
                            odd_even_sort<KeyType, ValueType, CompType, 2048>(shared, sharedValues, size, comp);
                        else
                            odd_even_sort<KeyType, ValueType, CompType, 2048, CTA_SIZE>(shared, sharedValues, size, comp);
                    } else if (size > 512) {
                        if (CTA_SIZE > 512)
                            odd_even_sort<KeyType, ValueType, CompType, 1024>(shared, sharedValues, size, comp);
                        else
                            odd_even_sort<KeyType, ValueType, CompType, 1024, CTA_SIZE>(shared, sharedValues, size, comp);
                    } else if (size > 256) {
                        if (CTA_SIZE > 256)
                            odd_even_sort<KeyType, ValueType, CompType, 512>(shared, sharedValues, size, comp);
                        else
                            odd_even_sort<KeyType, ValueType, CompType, 512, CTA_SIZE>(shared, sharedValues, size, comp);
                    } else if (size > 128) {
                        if (CTA_SIZE > 128)
                            odd_even_sort<KeyType, ValueType, CompType, 256>(shared, sharedValues, size, comp);
                        else
                            odd_even_sort<KeyType, ValueType, CompType, 256, CTA_SIZE>(shared, sharedValues, size, comp);
                    } else if (size > 64) {
                        if (CTA_SIZE > 64)
                            odd_even_sort<KeyType, ValueType, CompType, 128>(shared, sharedValues, size, comp);
                        else
                            odd_even_sort<KeyType, ValueType, CompType, 128, CTA_SIZE>(shared, sharedValues, size, comp);
                    } else if (size > 32) {
                        if (CTA_SIZE > 32)
                            odd_even_sort<KeyType, ValueType, CompType, 64>(shared, sharedValues, size, comp);
                        else
                            odd_even_sort<KeyType, ValueType, CompType, 64, CTA_SIZE>(shared, sharedValues, size, comp);
                    } else
                        odd_even_sort<KeyType, ValueType, CompType, 32>(shared, sharedValues, size, comp);

                    for (int i = threadIdx.x; i < size; i += CTA_SIZE) {
                        keysInput[i + from] = shared[i];
                        valuesInput[i + from] = sharedValues[i];
                    }
                }

                if (threadIdx.x == 0)
                    --stack;

                __syncthreads();
            }
                // Else do a splitting step.
            else {
                if (threadIdx.x == 0) {
                    KeyType mip = SampleSort::min<KeyType>
                                  (SampleSort::min<KeyType>(keys[from], keys[to - 1]), keys[(from + to) / 2]);
                    KeyType map = SampleSort::max<KeyType>
                                  (SampleSort::max<KeyType>(keys[from], keys[to - 1]), keys[(from + to) / 2]);
                    pivot = SampleSort::min<KeyType> (SampleSort::max < KeyType > (mip / 2 + map / 2, mip), map);
                }

                unsigned int ll = 0;
                unsigned int lm = 0;
                unsigned int lr = 0;

                __syncthreads();

                unsigned int coal = (from) & 0xf;

                if (threadIdx.x + from - coal < to) {
                    KeyType d = keys[threadIdx.x + from - coal];

                    if (threadIdx.x >= coal) {
                        if (comp(d, pivot)) ++ll;
                        else if (comp(pivot, d)) ++lr;
                        else ++lm;
                    }
                }

                for (int i = from + threadIdx.x + CTA_SIZE - coal; i < to; i += CTA_SIZE) {
                    KeyType d = keys[i];

                    if (comp(d, pivot)) ++ll;
                    else if (comp(pivot, d)) ++lr;
                    else ++lm;
                }

                // Store the result in shared memory for the prefix sums.
                smallBlock[threadIdx.x] = ll;
                equalBlock[threadIdx.x] = lm;
                largeBlock[threadIdx.x] = lr;

                __syncthreads();

                // Blelloch's scan is not optimal, but it is in-place.
                scan3<unsigned int, CTA_SIZE>(smallBlock, equalBlock, largeBlock);

                // Refill the stack.
                if (threadIdx.x == 0) {
                    flip[stack + 1] = !flip[stack];
                    flip[stack] = !flip[stack];

                    // Tail recursion to limit the stack size.
                    if (smallBlock[CTA_SIZE] < largeBlock[CTA_SIZE]) {
                        start[stack + 1] = from;
                        start[stack] = to - largeBlock[CTA_SIZE];
                        end[stack + 1] = from + smallBlock[CTA_SIZE];
                    } else {
                        end[stack + 1] = to;
                        end[stack] = from + smallBlock[CTA_SIZE];
                        start[stack + 1] = to - largeBlock[CTA_SIZE];
                    }

                    stack++;
                }

                __syncthreads();

                unsigned int x = from + smallBlock[threadIdx.x + 1] - 1;
                unsigned int y = to - largeBlock[threadIdx.x + 1];
                unsigned int z = from + smallBlock[CTA_SIZE] + equalBlock[threadIdx.x];

                // Use the prefix sums to write the elements to their new positions.
                if (threadIdx.x + from - coal < to) {
                    KeyType d = keys[threadIdx.x + from - coal];
                    ValueType vd = values[threadIdx.x + from - coal];

                    if (threadIdx.x >= coal) {
                        if (comp(d, pivot)) {
                            keys2[x] = d;
                            values2[x] = vd;
                            --x;
                        } else if (comp(pivot, d)) {
                            keys2[y] = d;
                            values2[y] = vd;
                            ++y;
                        } else {
                            keys2[z] = d;
                            values2[z] = vd;
                            ++z;
                        }
                    }
                }

                for (int i = from + threadIdx.x + CTA_SIZE - coal; i < to; i += CTA_SIZE) {
                    KeyType d = keys[i];
                    ValueType vd = values[i];

                    if (comp(d, pivot)) {
                        keys2[x] = d;
                        values2[x] = vd;
                        --x;
                    } else if (comp(pivot, d)) {
                        keys2[y] = d;
                        values2[y] = vd;
                        ++y;
                    } else {
                        keys2[z] = d;
                        values2[z] = vd;
                        ++z;
                    }
                }
                __syncthreads();

                // Write the keys/values equal to the pivot to the output array since
                // their final position is known.
                for (int i = from + smallBlock[CTA_SIZE] + threadIdx.x; i < to - largeBlock[CTA_SIZE]; i += CTA_SIZE) {
                    keysInput[i] = keys2[i];
                    valuesInput[i] = values2[i];
                }
                __syncthreads();
            }
        }
    }
}