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

#pragma once

namespace SampleSort {

    // Batcher's odd-even-merge-sort. Sorts elements in shared memory. PADDED_SIZE is the next
    // power of two larger than the input size.
    template<typename KeyType, typename CompType, int PADDED_SIZE, unsigned int CTA_SIZE>
    __device__ void odd_even_sort(KeyType *keys, const int size, CompType comp) {
        for (int p = PADDED_SIZE >> 1; p > 0; p >>= 1) {
            int r = 0, d = p;

            for (int q = PADDED_SIZE >> 1; q >= p; q >>= 1) {
                for (int k = threadIdx.x; k < size; k += CTA_SIZE) {
                    if (((k & p) == r) && ((k + d) < size)) {
                        KeyType sk = keys[k];
                        KeyType skd = keys[k + d];

                        if (comp(skd, sk)) {
                            keys[k] = skd;
                            keys[k + d] = sk;
                        }
                    }
                }

                d = q - p;
                r = p;

                __syncthreads();
            }
        }
    }

    // Sort elements in shared memory where the number of threads is not smaller
    // than the number of elements. Because of this the inner loop can be omitted,
    // which makes a measurable difference in running time. This is based on the
    // implementation from the Thrust library.
    template<typename KeyType, typename CompType, int PADDED_SIZE>
    __device__ void odd_even_sort(KeyType *keys, const int size, CompType comp) {
        for (int p = PADDED_SIZE >> 1; p > 0; p >>= 1) {
            int r = 0, d = p;

            for (int q = PADDED_SIZE >> 1; q >= p; q >>= 1) {
                int j = threadIdx.x + d;

                if ((threadIdx.x & p) == r && j < size) {
                    KeyType sk = keys[threadIdx.x];
                    KeyType skd = keys[j];

                    if (comp(skd, sk)) {
                        keys[threadIdx.x] = skd;
                        keys[j] = sk;
                    }
                }

                d = q - p;
                r = p;

                __syncthreads();
            }
        }
    }

    // Same as above but for key-value-pairs.
    template<typename KeyType, typename ValueType, typename CompType, int PADDED_SIZE, unsigned int CTA_SIZE>
    __device__ void odd_even_sort(KeyType *keys, ValueType *values, const int size, CompType comp) {
        for (int p = PADDED_SIZE >> 1; p > 0; p >>= 1) {
            int r = 0, d = p;

            for (int q = PADDED_SIZE >> 1; q >= p; q >>= 1) {
                for (int k = threadIdx.x; k < size; k += CTA_SIZE) {
                    if (((k & p) == r) && ((k + d) < size)) {
                        KeyType sk = keys[k];
                        KeyType skd = keys[k + d];

                        if (comp(skd, sk)) {
                            keys[k] = skd;
                            keys[k + d] = sk;

                            ValueType vk = values[k];
                            values[k] = values[k + d];
                            values[k + d] = vk;
                        }
                    }
                }

                d = q - p;
                r = p;

                __syncthreads();
            }
        }
    }

    // Same as above but for key-value-pairs.
    template<typename KeyType, typename ValueType, typename CompType, int PADDED_SIZE>
    __device__ void odd_even_sort(KeyType *keys, ValueType *values, const int size, CompType comp) {
        for (int p = PADDED_SIZE >> 1; p > 0; p >>= 1) {
            int q = PADDED_SIZE >> 1, r = 0, d = p;

            while (q >= p) {
                int j = threadIdx.x + d;

                if ((threadIdx.x & p) == r && j < size) {
                    KeyType sk = keys[threadIdx.x];
                    KeyType skd = keys[j];

                    if (comp(skd, sk)) {
                        keys[threadIdx.x] = skd;
                        keys[j] = sk;

                        ValueType vk = values[threadIdx.x];
                        values[threadIdx.x] = values[j];
                        values[j] = vk;
                    }
                }

                d = q - p;
                q >>= 1;
                r = p;

                __syncthreads();
            }
        }
    }
}
