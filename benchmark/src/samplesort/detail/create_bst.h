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

#include "bucket.h"
#include "scan.h"
#include "odd_even_sort.h"
#include "predicates.h"
#include "Lrand48.inl"
#include "constants.h"

namespace SampleSort {
    // For each CTA, sort a random sample of it's part of the input sequence and create a binary search tree.
    template<int K, int A, int CTA_SIZE, int LOCAL_SORT_SIZE, typename KeyType, typename CompType>
    __global__ static void create_bst(
            KeyType *keys_input,
            KeyType *keys_output,
            struct Bucket *buckets,
            KeyType *bst,
            KeyType *sample,
            KeyType *sample_buffer,
            Lrand48 rng,
            CompType comp) {
        __shared__ int from;
        __shared__ int size;
        __shared__ KeyType *input;
        __shared__ int block;

        if (threadIdx.x == 0) {
            block = blockIdx.x;
            from = buckets[blockIdx.x].start;
            size = buckets[blockIdx.x].size;
            if (!buckets[blockIdx.x].flipped)
                input = keys_input;
            else
                input = keys_output;
        }

        __syncthreads();

        lrand48LoadState(rng);

        for (int i = block * (A * K) + threadIdx.x; i < (block + 1) * (A * K); i += CTA_SIZE) {
            int randPos = from + (lrand48NextInt(rng) % size);
            sample[i] = input[randPos];
        }

        lrand48StoreState(rng);

        /// QUICKSORT.
        const int sharedSize = LOCAL_SORT_SIZE < 2 * CTA_SIZE ? 2 * CTA_SIZE : LOCAL_SORT_SIZE;
        // Used for shared memory sorting and for holding 3 prefix sum arrays of block size.
        __shared__
        KeyType shared[sharedSize];
        __shared__ unsigned int *buffer;

        // A stack is used to handle recursion.
        __shared__ int stack;
        __shared__ unsigned int start[32];
        __shared__ unsigned int end[32];
        __shared__ bool flip[32];

        // The total number of small and large elements.
        __shared__ unsigned int small_offset;
        __shared__ unsigned int large_offset;

        // The current pivot.
        __shared__ KeyType pivot;

        // The current sequence to be split.
        __shared__ int to;

        __shared__ KeyType *keys;
        __shared__ KeyType *keys2;

        // Initialize by putting the whole input on the stack.
        if (threadIdx.x == 0) {
            buffer = (unsigned int *) shared;
            start[0] = block * (A * K);
            end[0] = (block + 1) * (A * K);
            flip[0] = false;

            stack = 0;
        }

        __syncthreads();

        while (stack >= 0) {
            // Get the next sequence.
            if (threadIdx.x == 0) {
                from = start[stack];
                to = end[stack];
                size = to - from;

                keys = flip[stack] ? sample_buffer : sample;
                keys2 = flip[stack] ? sample : sample_buffer;
            }

            __syncthreads();

            // If the sequence is small enough it is sorted in shared memory.
            if (size <= LOCAL_SORT_SIZE) {
                if (size > 0) {
                    KeyType *shared_keys = (KeyType *) shared;
                    unsigned int coal = (from) & 0xf;
                    for (int i = threadIdx.x - coal; i < size; i += CTA_SIZE)
                        if (i >= 0) shared_keys[i] = keys[i + from];

                    __syncthreads();

                    // Depending on the size of the sequence to sort, choose a fitting permutation of odd-even-merge-sort.
                    // This case-distinction saves a few expensive instructions at run time. For sort sizes > 2048 more case
                    // distinctions would have to be added. Also, a minimum block size of 256 is assumed here.
                    if (size > 1024) {
                        if (CTA_SIZE > 1024)
                            odd_even_sort<KeyType, CompType, 2048>(shared_keys, size, comp);
                        else odd_even_sort<KeyType, CompType, 2048, CTA_SIZE>(shared_keys, size, comp);
                    } else if (size > 512) {
                        if (CTA_SIZE > 512) odd_even_sort<KeyType, CompType, 1024>(shared_keys, size, comp);
                        else odd_even_sort<KeyType, CompType, 1024, CTA_SIZE>(shared_keys, size, comp);
                    } else if (size > 256) {
                        if (CTA_SIZE > 256) odd_even_sort<KeyType, CompType, 512>(shared_keys, size, comp);
                        else odd_even_sort<KeyType, CompType, 512, CTA_SIZE>(shared_keys, size, comp);
                    } else if (size > 128) {
                        if (CTA_SIZE > 128) odd_even_sort<KeyType, CompType, 256>(shared_keys, size, comp);
                        else odd_even_sort<KeyType, CompType, 256, CTA_SIZE>(shared_keys, size, comp);
                    } else if (size > 64) {
                        if (CTA_SIZE > 64) odd_even_sort<KeyType, CompType, 128>(shared_keys, size, comp);
                        else odd_even_sort<KeyType, CompType, 128, CTA_SIZE>(shared_keys, size, comp);
                    } else if (size > 32) {
                        if (CTA_SIZE > 32) odd_even_sort<KeyType, CompType, 64>(shared_keys, size, comp);
                        else odd_even_sort<KeyType, CompType, 64, CTA_SIZE>(shared_keys, size, comp);
                    } else odd_even_sort<KeyType, CompType, 32>(shared_keys, size, comp);

                    for (int i = threadIdx.x; i < size; i += CTA_SIZE)
                        sample[i + from] = shared_keys[i];
                }

                if (threadIdx.x == 0)
                    --stack;

                __syncthreads();
            } else {
                if (threadIdx.x == 0) {
                    int middle = (from + to) / 2;
                    KeyType mip = SampleSort::min<KeyType>
                            (SampleSort::min<KeyType>(keys[from], keys[to - 1], comp), keys[middle], comp);
                    KeyType map = SampleSort::max<KeyType>
                            (SampleSort::max<KeyType>(keys[from], keys[to - 1], comp), keys[middle], comp);
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

                for (unsigned int i = from + threadIdx.x + CTA_SIZE - coal; i < to; i += CTA_SIZE) {
                    KeyType d = keys[i];

                    if (comp(d, pivot)) ++ll;
                    else if (comp(pivot, d)) ++lr;
                }

                // Generate offsets for writing small/large elements for each thread.
                buffer[threadIdx.x] = ll;
                __syncthreads();
                brent_kung_inclusive<unsigned int, CTA_SIZE>(buffer);

                unsigned int x = from + buffer[threadIdx.x + 1] - 1;
                __syncthreads();
                if (threadIdx.x == 0) small_offset = buffer[CTA_SIZE];
                buffer[threadIdx.x] = lr;
                __syncthreads();
                brent_kung_inclusive<unsigned int, CTA_SIZE>(buffer);

                unsigned int y = to - buffer[threadIdx.x + 1];

                if (threadIdx.x == 0) {
                    large_offset = buffer[CTA_SIZE];

                    // Refill the stack.
                    flip[stack + 1] = !flip[stack];
                    flip[stack] = !flip[stack];

                    if (small_offset < large_offset) {
                        start[stack + 1] = start[stack];
                        start[stack] = to - large_offset;
                        end[stack + 1] = from + small_offset;
                    } else {
                        end[stack + 1] = end[stack];
                        end[stack] = from + small_offset;
                        start[stack + 1] = to - large_offset;
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

                for (unsigned int i = from + threadIdx.x + CTA_SIZE - coal; i < to; i += CTA_SIZE) {
                    KeyType d = keys[i];

                    if (comp(d, pivot)) keys2[x--] = d;
                    else if (comp(pivot, d)) keys2[y++] = d;
                }

                __syncthreads();

                // Write the keys equal to the pivot to the output array since
                // their final position is known.
                for (unsigned int i = from + small_offset + threadIdx.x; i < to - large_offset; i += CTA_SIZE)
                    sample[i] = pivot;
                __syncthreads();
            }
        }
        /// END QUICKSORT.

        if (threadIdx.x == 0) {
            KeyType first = sample[block * (A * K) + A];
            KeyType last = sample[block * (A * K) + A * (K - 1)];

            // If all splitters are equal do not bother to create the bst.
            if (!comp(first, last) && !comp(last, first)) {
                buckets[block].degenerated = true;
                bst[K * block] = sample[block * (A * K) + A];
            }
        }
        __syncthreads();
        if (!buckets[block].degenerated) {
            for (int i = threadIdx.x; i < K; i += CTA_SIZE) {
                int _l = (int) __log2f(i + 1);
                int pos = (A * K * (1 + 2 * ((i + 1) & (1 << _l) - 1))) / (2 << _l);
                bst[i + K * block] = sample[block * (A * K) + pos];
            }
        }
    }
}
