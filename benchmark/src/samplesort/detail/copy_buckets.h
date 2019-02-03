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

#include "bucket.h"

namespace SampleSort {
    // Each CTA copies the data of one bucket from the buffer to the input array.
    template<int CTA_SIZE, typename KeyType>
    __global__ void copy_buckets(
            KeyType *keys,
            KeyType *keys_buffer,
            struct Bucket *buckets) {
        const int from = buckets[blockIdx.x].start;
        const int to = from + buckets[blockIdx.x].size;

        for (int i = threadIdx.x + from; i < to; i += CTA_SIZE)
            keys[i] = keys_buffer[i];
    }

    template<int CTA_SIZE, typename KeyType, typename ValueType>
    __global__ void copy_buckets(
            KeyType *keys,
            KeyType *keys_buffer,
            ValueType *values,
            ValueType *values_buffer,
            struct Bucket *buckets) {
        const int from = buckets[blockIdx.x].start;
        const int to = from + buckets[blockIdx.x].size;

        for (int i = threadIdx.x + from; i < to; i += CTA_SIZE) {
            keys[i] = keys_buffer[i];
            values[i] = values_buffer[i];
        }
    }
}
