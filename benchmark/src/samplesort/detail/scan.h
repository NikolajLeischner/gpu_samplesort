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

    // Inclusive scan, shifted one to the right. Requires n additional space.
    // Works for powers of 2 that fit into shared memory and are >=64.
    // This is like the Brent-Kung implementation described in
    // "Parallel Scan for Stream Architectures", and could be optimized further
    // by using a depth-minimal scan algorithm for intra-warp scans.
    template<typename T, int n>
    __device__ void brent_kung_inclusive(T *data) {
        T *a = data;
        int id = threadIdx.x;

        int d = n;
        for (; d > 64; d /= 2) {
            if (id < d / 2) a[d + id] = a[2 * id] + a[2 * id + 1];
            a += d;
            __syncthreads();
        }

        for (; d > 2; d /= 2) {
            if (id < d / 2) a[d + id] = a[2 * id] + a[2 * id + 1];
            a += d;
        }
        __syncthreads();

        if (id == 0) {
            a[2] = a[1] + a[0];
            a[1] = a[0];
            a[0] = 0;
        }
        __syncthreads();

        d = 4;
        for (; d <= 64; d *= 2) {
            a -= d;
            if (id < d / 2) {
                a[2 * id + 1] = a[2 * id] + a[d + id];
                a[2 * id] = a[d + id];
            }
        }
        __syncthreads();

        for (; d <= n; d *= 2) {
            a -= d;
            if (id < d / 2) {
                a[2 * id + 1] = a[2 * id] + a[d + id];
                a[2 * id] = a[d + id];
            }
            __syncthreads();
        }

        if (id == 0) a[n] = a[2 * n - 2];
        __syncthreads();
    }

    // Inclusive scan, shifted one position to the right (the first position's value is zero).
    // Works on 3 arrays of block size. Based on the implementation from Daniel Cederman
    // and Philippas Tsigas GPU quicksort code.
    template<typename T, unsigned int CTA_SIZE>
    __device__ void scan3(T *lblock, T *rblock, T *mblock) {
        int offset = 1;

        for (int d = CTA_SIZE >> 1; d > 0; d >>= 1) {

            if (threadIdx.x < d) {
                int ai = offset * ((threadIdx.x << 1) + 1) - 1;
                int bi = offset * ((threadIdx.x << 1) + 2) - 1;

                lblock[bi] += lblock[ai];
                rblock[bi] += rblock[ai];
                mblock[bi] += mblock[ai];
            }
            offset <<= 1;
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            lblock[CTA_SIZE] = lblock[CTA_SIZE - 1];
            rblock[CTA_SIZE] = rblock[CTA_SIZE - 1];
            mblock[CTA_SIZE] = mblock[CTA_SIZE - 1];
            lblock[CTA_SIZE - 1] = 0;
            rblock[CTA_SIZE - 1] = 0;
            mblock[CTA_SIZE - 1] = 0;
        }
        __syncthreads();

        for (int d = 1; d < CTA_SIZE; d *= 2) {
            offset >>= 1;

            if (threadIdx.x < d) {
                int ai = offset * ((threadIdx.x << 1) + 1) - 1;
                int bi = offset * ((threadIdx.x << 1) + 2) - 1;

                T t = lblock[ai];
                lblock[ai] = lblock[bi];
                lblock[bi] += t;

                t = rblock[ai];
                rblock[ai] = rblock[bi];
                rblock[bi] += t;

                t = mblock[ai];
                mblock[ai] = mblock[bi];
                mblock[bi] += t;
            }
            __syncthreads();
        }
    }
}
