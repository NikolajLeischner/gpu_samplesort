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

#include "../SampleSort.h"

namespace SampleSort
{
  // Create bucket counters which are relative to the CTA. To obtain global
  // offsets for scattering a prefix sum has to be performed afterwards.
  template<typename KeyType, typename StrictWeakOrdering, int K, int LOG_K, int CTA_SIZE, int COUNTERS, int COUNTER_COPIES, bool DEGENERATED>
  __global__ static void globalFindBuckets(KeyType *keys,	int minPos,	int maxPos,	int *globalBuckets,	
    int elementsPerThread, // The number of keys each thread processes. I.e. the number of CTAs required for buckets of 
    // different sizes can be adjusted. For one bucket the bucket-finding and scattering kernel must use the same value.
    StrictWeakOrdering comp) 			
  {		
    const int LOCAL_COUNTERS = COUNTERS * COUNTER_COPIES;
    // This reduces register usage.
    __shared__ int block;
    __shared__ int grid;
    if (threadIdx.x == 0) 
    {
      block = blockIdx.x;
      grid = gridDim.x;
    }
    __syncthreads();

    const int findBlockElements = elementsPerThread * CTA_SIZE;
    const int from = block * findBlockElements + minPos;
    int to = (block + 1 == grid) ? maxPos: from + findBlockElements;

    __shared__ KeyType bst[K];
    __shared__ int buckets[K * LOCAL_COUNTERS];
    
    KeyType *constBst = reinterpret_cast<KeyType*>(bstCache);	

    for (int i = threadIdx.x; i < K * LOCAL_COUNTERS; i += CTA_SIZE) buckets[i] = 0;

    if (!DEGENERATED)
    {
      for (int i = threadIdx.x; i < K; i += CTA_SIZE) bst[i] = constBst[i];
    }    
    // All splitters for the bucket are identical, don't even load the bst but just one splitter.
    else if (threadIdx.x == 0) bst[0] = constBst[0];
    __syncthreads();

    for (int i = from + threadIdx.x; i < to; i += CTA_SIZE)
    {
        int bucketPos = 1;
        KeyType d = keys[i];

        if (!DEGENERATED)
        {
          // Traverse bst.
          for (int j = 0; j < LOG_K; ++j)
          {
            if (comp(bst[bucketPos - 1], d)) bucketPos = (bucketPos << 1) + 1;
            else                             bucketPos <<= 1;
          }

          bucketPos = bucketPos - K;
        }
        else
        {
          if      (comp(bst[0], d)) bucketPos = 2;
          else if (comp(d, bst[0])) bucketPos = 0;
        }

        atomicAdd(buckets + bucketPos * LOCAL_COUNTERS + threadIdx.x % LOCAL_COUNTERS, 1);
    }

    __syncthreads();

    // Sum up and write back CTA bucket counters.
    for (int i = threadIdx.x; i < K; i += CTA_SIZE) 
    {
      for (int j = 0; j < COUNTERS; ++j)
      {
        int b = 0;
        for (int k = 0; k < COUNTER_COPIES; ++k)
          b += buckets[i * LOCAL_COUNTERS + j + k * COUNTERS];

        globalBuckets[(i * grid * COUNTERS) + (block * COUNTERS) + j] = b;
      }		
    }
  }
}