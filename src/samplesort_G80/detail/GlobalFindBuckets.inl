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
  template<typename KeyType, typename StrictWeakOrdering, int K, int CTA_SIZE, int COUNTERS, int COUNTER_COPIES, bool DEGENERATED>
  __global__ void globalFindBuckets(KeyType *keys,	int minPos,	int maxPos,	int *globalBuckets,	
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

    const int from = block * elementsPerThread * CTA_SIZE + minPos;
    const int to = block + 1 == grid ? maxPos: from + elementsPerThread * CTA_SIZE;

    // Shared memory copy of the search tree.
    __shared__ KeyType bst[K];
    __shared__ int bucketPos[CTA_SIZE];
    __shared__ int buckets[K * LOCAL_COUNTERS];
    
    KeyType *constBst = reinterpret_cast<KeyType*>(bstCache);	

    for (int i = threadIdx.x; i < (K * LOCAL_COUNTERS); i += CTA_SIZE) buckets[i] = 0;

    if (!DEGENERATED)
    {
      for (int i = threadIdx.x; i < K; i += CTA_SIZE) bst[i] = constBst[i];
    }    
    // All splitters for the bucket are identical, don't even load the
    // bst but just one splitter.
    else  if (threadIdx.x == 0) bst[0] = constBst[0];
    __syncthreads();

    // Instead of loading one key in each iteration, load two in each second iteration.
    // This allows the hardware to pipeline the load requests better.
    KeyType d1, d2;
    bool even = true;
    for (int i = from; i < to; i += CTA_SIZE)
    {
      int numElems = min(to - i, CTA_SIZE);

      if (threadIdx.x < numElems)
      {
	  	 if (even) 
        {
          d1 = keys[i + threadIdx.x];
          if (i + threadIdx.x + CTA_SIZE < to) d2 = keys[i + threadIdx.x + CTA_SIZE];
        }
		
        KeyType d = even ? d1: d2;
        even = !even;
        bucketPos[threadIdx.x] = 1;

        if (!DEGENERATED)
        {
          // Traverse bst.
          for (int j = 0; j < lg<K>::result; ++j)
          {
            if (comp(bst[bucketPos[threadIdx.x] - 1], d))
              bucketPos[threadIdx.x] = (bucketPos[threadIdx.x] << 1) + 1;
            else bucketPos[threadIdx.x] <<= 1;
          }

          bucketPos[threadIdx.x] = bucketPos[threadIdx.x] - K;
        }
        else
        {
          if      (comp(bst[0], d)) bucketPos[threadIdx.x] = 2;
          else if (comp(d, bst[0])) bucketPos[threadIdx.x] = 0;
        }
      }

      __syncthreads();
      // Only the first COUNTERS threads increment bucket counters
      // to avoid conflicts when using shared memory.
        if (threadIdx.x < LOCAL_COUNTERS)
        {
          for (int j = threadIdx.x; j < numElems; j += LOCAL_COUNTERS)
            ++buckets[bucketPos[j] * LOCAL_COUNTERS + threadIdx.x];
        }
      __syncthreads();
    }

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