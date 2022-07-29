/**
* This is a linear congruential generator with the same output as
* lrand48() from the C standard library. Refer to Donald Knuth, 
* The Art of Computer Programming, Volume 2, Section 3.2.1. for
* more information on the topic of pseudo-random number generation.
**/

#pragma once

namespace SampleSort {
    struct Lrand48 {
        uint2 A;
        uint2 C;
        uint2 *state;
        uint2 state0;

        static const unsigned long long a = 0x5DEECE66DLL;
        static const unsigned long long c = 0xB;

        void init(int threads, int seed) {
            auto *seeds = new uint2[threads];

            cudaMalloc((void **) &state, threads * sizeof(uint2));

            unsigned long long A, C;
            A = 1LL;
            C = 0LL;

            for (int i = 0; i < threads; ++i) {
                C += A * c;
                A *= a;
            }

            this->A.x = A & 0xFFFFFFLL;
            this->A.y = (A >> 24) & 0xFFFFFFLL;
            this->C.x = C & 0xFFFFFFLL;
            this->C.y = (C >> 24) & 0xFFFFFFLL;

            unsigned long long x = (((unsigned long long) seed) << 16) | 0x330E;
            for (int i = 0; i < threads; ++i) {
                x = a * x + c;
                seeds[i].x = x & 0xFFFFFFLL;
                seeds[i].y = (x >> 24) & 0xFFFFFFLL;
            }

            cudaMemcpy(state, seeds, threads * sizeof(uint2), cudaMemcpyHostToDevice);

            delete[] seeds;
        }

        void destroy() const {
            cudaFree((void *) state);
        }
    };

    __device__ inline void lrand48LoadState(Lrand48 &r) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        r.state0 = r.state[i];
    }

    __device__ inline void lrand48StoreState(Lrand48 &r) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        r.state[i] = r.state0;
    }

    __device__ inline void lrand48LoadStateSimple(Lrand48 &r) {
        r.state0 = r.state[threadIdx.x];
    }

    __device__ inline void lrand48StoreStateSimple(Lrand48 &r) {
        r.state[threadIdx.x] = r.state0;
    }

    __device__ inline void lrand48Iterate(Lrand48 &r) {

        const unsigned int low = __umul24(r.state0.x, r.A.x);
        const unsigned int high = __umulhi(r.state0.x, r.A.x);

        unsigned int R0 = ((low & 0xFFFFFF) + r.C.x) & 0xFFFFFF;

        unsigned int R1 = (((high >> 24) | (high << 8)) + r.C.y + (R0 >> 24) +
                           __umul24(r.state0.y, r.A.x) + __umul24(r.state0.x, r.A.y)) & 0xFFFFFF;

        r.state0 = make_uint2(R0, R1);
    }

    __device__ inline unsigned int lrand48NextInt(Lrand48 &r) {
        unsigned int res = (r.state0.x >> 17) | (r.state0.y << 7);
        lrand48Iterate(r);
        return res;
    }

    // Returns a float in the range [0, 1).
    __device__ inline float lrand48NextFloat(Lrand48 &r) {
        float res = r.state0.y / (float) (1 << 24);
        lrand48Iterate(r);
        return res;
    }
}
