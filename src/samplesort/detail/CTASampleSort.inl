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
#include "Bucket.inl"

namespace SampleSort
{
  /* Block-wise sample sort. Each block sorts a sequence by itself, without cooperation between blocks. */
  template<typename KeyType, typename StrictWeakOrdering, int LOCAL_SORT_SIZE, int CTA_SIZE>
  __global__ void CTASampleSort(KeyType *keysInput, KeyType *keysBuffer, struct Bucket *bucketParams,
    Lrand48 rng, StrictWeakOrdering comp)
  {
    lrand48LoadStateSimple(rng); 
    const int QUICKSORT_SIZE = 1 << 13;
    // Number of data scattering threads, should be <= 16.
    const int NUM_SCATTER = 16; 
    // Oversampling factor. 
    const int A = 6; 
    // Next larger power of 2.
    const int A_P2 = 8;
    // Search tree size. Must be a power of 2.
    const int K = 32; 
    // Size of the stack for large buckets.
    const int STACK_SIZE = 64; 

    // Temporary storage.
    __shared__ KeyType shared[LOCAL_SORT_SIZE];

    // Sample sort.

    // Binary search tree.
    __shared__ KeyType bst[K];

    // Bucket positions from the last k-way splitting.
    __shared__ int newBucketsStart[K + 1]; 
    __shared__ int newStack; 

    // Stack for large buckets.
    __shared__ int start[STACK_SIZE]; 
    __shared__ int end[STACK_SIZE]; 
    __shared__ bool largeBucketsFlip[STACK_SIZE]; 
    __shared__ int largeStack;

    // Pointers for reading & writing data, as we alternate with each iteration.
    __shared__ bool flip; 
    __shared__ KeyType *keys;
    __shared__ KeyType *keys2;

    // Flag indicating if during the previous split all splitters were equal.
    __shared__ bool splitConstant;
    __shared__ KeyType lSplit; 
    __shared__ int size;
    __shared__ int from;
    __shared__ int to;

    // Quicksort.

    // Stack.
    __shared__ bool wasFlipped;
    __shared__ int qsStack;
    __shared__ int qsStart[8];
    __shared__ int qsEnd[8];
    __shared__ bool qsFlip[8];

    // The total number of small and large elements.
    __shared__ unsigned int smallOffset;
    __shared__ unsigned int largeOffset;

    // The current pivot.
    __shared__ KeyType pivot;

    // Initialization and control over sorting is done by thread 0.
    if (threadIdx.x == 0)
    {
      flip = bucketParams[blockIdx.x].flipped;
      newStack = 0;
      newBucketsStart[0] = bucketParams[blockIdx.x].start;
      newBucketsStart[1] = bucketParams[blockIdx.x].start + bucketParams[blockIdx.x].size;
      largeStack = -1;
      splitConstant = false;

      keys = flip ? keysBuffer: keysInput;
      keys2 = flip ? keysInput: keysBuffer;
    }

    __syncthreads();

    while(largeStack  >= 0 || newStack >= 0)
    {
      while(newStack >= 0)
      {
        if (threadIdx.x == 0)
        {
          from = newBucketsStart[newStack];
          to = newBucketsStart[newStack + 1];
          size = to - from;
        }
        __syncthreads();

        // Bucket is empty or contains only equal keys. Just move it to the correct array, if necessary.
        if ((size == 1) || ((newStack == 1) && (splitConstant)))
        {
          if (flip)
          {
            for (int i = from + threadIdx.x; i < to; i += CTA_SIZE)
              keysInput[i] = keys[i];
          }
        }
        // Sort bucket in shared memory and copy back to global memory.
        else if (size < QUICKSORT_SIZE)
        {
          if (threadIdx.x == 0)
          {
            wasFlipped = flip;
            qsStart[0] = from;
            qsEnd[0] = to;
            qsFlip[0] = flip;
            qsStack = 0;
          }
          __syncthreads();

          while (qsStack >= 0)
          {
            // Get the next sequence.
            if (threadIdx.x == 0)
            {
              from = qsStart[qsStack];
              to = qsEnd[qsStack];
              size = to - from;

              keys = qsFlip[qsStack] ? keysBuffer: keysInput;
              keys2 = qsFlip[qsStack] ? keysInput: keysBuffer;
            }	

            __syncthreads();

            // If the sequence is small enough it is sorted in shared memory.
            if (size <= LOCAL_SORT_SIZE)
            {
              if (size > 0)
              {
                unsigned int coal = (from)&0xf;
                for (int i = threadIdx.x - coal; i < size; i += CTA_SIZE)
                  if (i >= 0) shared[i] = keys[i + from];

                __syncthreads();

                // Depending on the size of the sequence to sort, choose a fitting permutation of odd-even-merge-sort.
                // This case-distinction saves a few expensive instructions at run time. For sort sizes > 2048 more case 
                // distinctions would have to be added. Also a minimum block size of 256 is assumed here.
                if (size > 1024) 
                {
                  if (CTA_SIZE > 1024) oddEvenSortSmall<KeyType, StrictWeakOrdering, 2048>(shared, size, comp);
                  else oddEvenSort<KeyType, StrictWeakOrdering, 2048, CTA_SIZE>(shared, size, comp);
                }
                else if (size > 512)
                {
                  if (CTA_SIZE > 512) oddEvenSortSmall<KeyType, StrictWeakOrdering, 1024>(shared, size, comp);
                  else oddEvenSort<KeyType, StrictWeakOrdering, 1024, CTA_SIZE>(shared, size, comp);
                }
                else if (size > 256) 
                {
                  if (CTA_SIZE > 256) oddEvenSortSmall<KeyType, StrictWeakOrdering, 512>(shared, size, comp);
                  else oddEvenSort<KeyType, StrictWeakOrdering, 512, CTA_SIZE>(shared, size, comp);
                }
                else if (size > 128)
                {
                  if (CTA_SIZE > 128) oddEvenSortSmall<KeyType, StrictWeakOrdering, 256>(shared, size, comp);
                  else oddEvenSort<KeyType, StrictWeakOrdering, 256, CTA_SIZE>(shared, size, comp);
                }
                else if (size > 64)
                {
                  if (CTA_SIZE > 64) oddEvenSortSmall<KeyType, StrictWeakOrdering, 128>(shared, size, comp);
                  else oddEvenSort<KeyType, StrictWeakOrdering, 128, CTA_SIZE>(shared, size, comp);
                }
                else if (size > 32)
                {
                  if (CTA_SIZE > 32) oddEvenSortSmall<KeyType, StrictWeakOrdering, 64>(shared, size, comp);
                  else oddEvenSort<KeyType, StrictWeakOrdering, 64, CTA_SIZE>(shared, size, comp);
                }
                else oddEvenSortSmall<KeyType, StrictWeakOrdering, 32>(shared, size, comp);

                for (int i = threadIdx.x; i < size; i += CTA_SIZE)
                  keysInput[i + from] = shared[i];
              }

              if (threadIdx.x == 0) 
                --qsStack;

              __syncthreads();
            }
            else
            {
              if (threadIdx.x == 0)
              {
                KeyType mip = SampleSort::min<KeyType>(SampleSort::min<KeyType>(keys[from], keys[to - 1], comp), keys[(from + to) / 2], comp);
                KeyType map = SampleSort::max<KeyType>(SampleSort::max<KeyType>(keys[from], keys[to - 1], comp), keys[(from + to) / 2], comp);
                pivot = SampleSort::min<KeyType>(SampleSort::max<KeyType>(mip / 2 + map / 2, mip, comp), map, comp);
              }

              unsigned int ll = 0;
              unsigned int lr = 0;

              unsigned int coal = (from)&0xf;

              __syncthreads();

              if (threadIdx.x + from - coal < to)
              {
                KeyType d = keys[threadIdx.x + from - coal];

                if(threadIdx.x >= coal)
                {
                  if      (comp(d, pivot)) ++ll;
                  else if (comp(pivot, d)) ++lr;
                }
              }

              for (int i = from + threadIdx.x + CTA_SIZE - coal; i < to; i += CTA_SIZE)
              {
                KeyType d = keys[i];

                if      (comp(d, pivot)) ++ll;
                else if (comp(pivot, d)) ++lr;
              }

              // Generate offsets for writing small/large elements for each thread.
              unsigned int *buffer = (unsigned int*)shared;
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

              if (threadIdx.x == 0)
              {
                largeOffset = buffer[CTA_SIZE];

                // Refill the qsStack.
                qsFlip[qsStack+1] = !qsFlip[qsStack];
                qsFlip[qsStack] = !qsFlip[qsStack];

                if (smallOffset < largeOffset)
                {
                  qsStart[qsStack + 1] = qsStart[qsStack];
                  qsStart[qsStack] = to - largeOffset;
                  qsEnd[qsStack + 1] = from + smallOffset;
                }
                else
                {
                  qsEnd[qsStack + 1] = qsEnd[qsStack];
                  qsEnd[qsStack] = from + smallOffset;
                  qsStart[qsStack + 1] = to - largeOffset;
                }

                ++qsStack;
              }

              __syncthreads();


              // Use the offsets generated by the prefix sums to write the elements to their new positions.
              if (threadIdx.x + from - coal < to)
              {
                KeyType d = keys[threadIdx.x + from - coal];

                if (threadIdx.x >= coal)
                {
                  if      (comp(d, pivot)) keys2[x--] = d;
                  else if (comp(pivot, d)) keys2[y++] = d;
                }
              }

              for (int i = from + threadIdx.x + CTA_SIZE - coal; i < to; i += CTA_SIZE)
              {	
                KeyType d = keys[i];

                if      (comp(d, pivot)) keys2[x--] = d;
                else if (comp(pivot, d)) keys2[y++] = d;
              }

              __syncthreads();

              // Write the keys equal to the pivot to the output array since their final position is known.
              for (int i = from + smallOffset + threadIdx.x; i < to - largeOffset; i += CTA_SIZE)
                keysInput[i] = pivot;
              __syncthreads();
            }
          }

          if (threadIdx.x == 0)
          {
            flip = wasFlipped;
            keys = flip ? keysBuffer: keysInput;
            keys2 = flip ? keysInput: keysBuffer;
          }
        }
        // The bucket is too large, put it on the stack for k-way splitting.
        else
        {
          if (threadIdx.x == 0)
          {
            ++largeStack;
            start[largeStack] = from;
            end[largeStack] = to;
            largeBucketsFlip[largeStack] = flip;      
          }
        }

        if (threadIdx.x == 0)
          --newStack;

        __syncthreads();
      }

      __syncthreads();

      if (largeStack >= 0)
      {
        // Pop a large bucket from the stack, taking into account in which of the two keys buffers it resides.
        __syncthreads();
        if (threadIdx.x == 0)
        {
          from = start[largeStack];
          to = end[largeStack];
          size = to - from;
          flip = largeBucketsFlip[largeStack];
          keys = flip ? keysBuffer: keysInput;
          keys2 = flip ? keysInput: keysBuffer;
          --largeStack;
        }
        __syncthreads();

        // K-way split the smallest largest bucket.
        KeyType *sample = &shared[0];	
        for (int i = threadIdx.x; i < (A * K); i += CTA_SIZE)
        {
          int randPos = from + (lrand48NextInt(rng) % size);
          sample[i] = keys[randPos];
        }	
        __syncthreads();

        oddEvenSortSmall<KeyType, StrictWeakOrdering, A_P2 * K>(sample, A * K, comp);

        // Check if all splitters are equal, in which case no bst traversal will be done.
        if (threadIdx.x == 0)
        {
          lSplit = sample[A];
          KeyType hSplit = sample[A * (K - 1)];
          splitConstant = !comp(lSplit, hSplit) && !comp(hSplit, lSplit);
        }
        __syncthreads();

        if (!splitConstant)
        {
          // Create bst and store it in shared memory.
          for (int i = threadIdx.x; i < K; i += CTA_SIZE)
          {
            int l = (int)__log2f(i + 1);
            int pos = (A * K * (1 + 2 * ((i + 1) % (1 << l)))) / (2 << l);
            bst[i] = sample[pos];
          }	
        }

        __syncthreads();

        // Initialize the bucket counters.
        int *bucketCounters = (int*)&shared[0];
        for (int i = threadIdx.x; i < NUM_SCATTER * K; i += CTA_SIZE) bucketCounters[i] = 0;
        __syncthreads();

        // Each thread loads 1 element and traverses the search tree, until
        // all elements are done. This is done in batches so the bucket
        // positions can be kept in shared memory.
        if (!splitConstant)
        {
          for (int i = from + threadIdx.x; i < to; i += CTA_SIZE)
          { 
              KeyType d = keys[i];		
              int bucketPos = 1;

              // Traverse bst.
              for (int j = 0; j < lg<K>::result; ++j)
              {
                if (comp(bst[bucketPos - 1], d)) bucketPos = (bucketPos << 1) + 1;
                else                             bucketPos <<= 1;
              }

              bucketPos = bucketPos - K;
              atomicAdd(bucketCounters + bucketPos * NUM_SCATTER + threadIdx.x % NUM_SCATTER, 1);
          }
        }
        else
        {
          for (int i = from  + threadIdx.x; i < to; i += CTA_SIZE)
          {     
              KeyType d = keys[i];		
              int bucketPos = 1;

              if      (comp(lSplit, d)) bucketPos = 2;
              else if (comp(d, lSplit)) bucketPos = 0;    

              atomicAdd(bucketCounters + bucketPos * NUM_SCATTER + threadIdx.x % NUM_SCATTER, 1);
          }
        }

        __syncthreads();

        brent_kung_exclusive<int, K * NUM_SCATTER>(bucketCounters);

        // Copy the new bucket positions. If all splitters were equal, only
        // the first three bucket can be non-empty and have to go onto the stack.
        if (!splitConstant)
        {
          for (int i = threadIdx.x; i < K; i += CTA_SIZE)
            newBucketsStart[i] = from + bucketCounters[i * NUM_SCATTER];
        }
        else
        {
          for (int i = threadIdx.x; i < 3; i += CTA_SIZE)
            newBucketsStart[i] = from + bucketCounters[i * NUM_SCATTER];
        }

        // Find the bucket positions in batches of block size, now writing out elements each time.
        if (!splitConstant)
        {
          for (int i = from + threadIdx.x; i < to; i += CTA_SIZE)
          {       
              KeyType d = keys[i];
              int bucketPos = 1;

              // Traverse bst.
              for (int j = 0; j < lg<K>::result; ++j)
              {
                if (comp(bst[bucketPos - 1], d)) bucketPos = (bucketPos << 1) + 1;
                else                             bucketPos <<= 1;
              }		

              bucketPos = bucketPos - K;
              keys2[from + atomicAdd(bucketCounters + bucketPos * NUM_SCATTER + threadIdx.x % NUM_SCATTER, 1)] = d;
          }
        }
        else
        {
          for (int i = from + threadIdx.x; i < to; i += CTA_SIZE)
          {      
              KeyType d = keys[i];
              int bucketPos = 1;

              if      (comp(lSplit, d)) bucketPos = 2;
              else if (comp(d, lSplit)) bucketPos = 0;

              keys2[from + atomicAdd(bucketCounters + bucketPos * NUM_SCATTER + threadIdx.x % NUM_SCATTER, 1)] = d;
          }
        }

        __syncthreads();

        // Prepare for the next iteration.
        if (threadIdx.x == 0)
        {
          keys = flip ? keysInput: keysBuffer;
          keys2 = flip ? keysBuffer: keysInput;
          flip = !flip;

          // If all splitters were equal only the first 3 buckets have to be put onto the stack.
          if (!splitConstant)
          {
            newBucketsStart[K] = to;
            newStack = K - 1;
          }
          else
          {
            newBucketsStart[3] = to;
            newStack = 2;
          }
        }
        __syncthreads();
      }	
    }
    lrand48StoreStateSimple(rng); 
  }

  /* Block-wise key-value sample sort. Each block sorts a sequence by itself, without cooperation between blocks. */
  template<typename KeyType, typename ValueType, typename StrictWeakOrdering, int LOCAL_SORT_SIZE, unsigned int CTA_SIZE>
  __global__ void CTASampleSortKeyValue(KeyType *keysInput, KeyType *keysBuffer, ValueType *valuesInput, 
    ValueType *valuesBuffer, struct Bucket *bucketParams, Lrand48 rng, StrictWeakOrdering comp)
  {
    lrand48LoadStateSimple(rng);  
    const int QUICKSORT_SIZE = 1 << 13;
    // Number of data scattering threads, should be <= 16.
    const int NUM_SCATTER = 16; 
    // Oversampling factor. 
    const int A = 6; 
    // Next larger power of 2.
    const int A_P2 = 8;
    // Search tree size. Must be a power of 2.
    const int K = 32; 
    // Size of the stack for large buckets.
    const int STACK_SIZE = 64; 

    // Temporary storage.
    __shared__ KeyType shared[LOCAL_SORT_SIZE];
    __shared__ ValueType sharedValues[LOCAL_SORT_SIZE]; 

    // Sample sort.

    // Binary search tree.
    __shared__ KeyType bst[K];

    // Bucket positions from the last k-way splitting.
    __shared__ int newBucketsStart[K + 1]; 
    __shared__ int newStack; 

    // Stack for large buckets.
    __shared__ int start[STACK_SIZE]; 
    __shared__ int end[STACK_SIZE]; 
    __shared__ bool largeBucketsFlip[STACK_SIZE]; 
    __shared__ int largeStack;

    // Pointers for reading & writing data, as we alternate with each iteration.
    __shared__ bool flip; 
    __shared__ KeyType *keys;
    __shared__ KeyType *keys2;
    __shared__ ValueType *values;
    __shared__ ValueType *values2;

    // Flag indicating if during the previous split all splitters were equal.
    __shared__ bool splitConstant;
    __shared__ KeyType lSplit; 
    __shared__ int size;
    __shared__ int from;
    __shared__ int to;

    // Quicksort.

    // Stack.
    __shared__ bool wasFlipped;
    __shared__ int qsStack;
    __shared__ int qsStart[8];
    __shared__ int qsEnd[8];
    __shared__ bool qsFlip[8];

    // Arrays for calculating prefix sums over the number of smaller/equal/larger elements
    // for each thread.
    unsigned int *smallBlock = (unsigned int*)shared;
    unsigned int *equalBlock = (unsigned int*)(&smallBlock[(CTA_SIZE + 1)]);
    unsigned int *largeBlock = (unsigned int*)sharedValues;

    // The current pivot.
    __shared__ KeyType pivot;

    // Initialization and control over sorting is done by thread 0.
    if (threadIdx.x == 0)
    {
      flip = bucketParams[blockIdx.x].flipped;
      newStack = 0;
      newBucketsStart[0] = bucketParams[blockIdx.x].start;
      newBucketsStart[1] = bucketParams[blockIdx.x].start + bucketParams[blockIdx.x].size;
      largeStack = -1;
      splitConstant = false;

      if (!flip)
      {
        keys = keysInput;
        keys2 = keysBuffer;
        values = valuesInput;
        values2 = valuesBuffer;
      }
      else
      {
        keys = keysBuffer;
        keys2 = keysInput;
        values = valuesBuffer;
        values2 = valuesInput;
      }
    }

    __syncthreads();

    while(largeStack  >= 0 || newStack >= 0)
    {
      while(newStack >= 0)
      {
        if (threadIdx.x == 0)
        {
          from = newBucketsStart[newStack];
          to = newBucketsStart[newStack + 1];
          size = to - from;
        }
        __syncthreads();

        // Bucket is empty or contains only equal keys. Just move it to the correct array, if necessary.
        if ((size == 1) || ((newStack == 1) && (splitConstant)))
        {
          if (flip)
          {
            for (int i = from + threadIdx.x; i < to; i += CTA_SIZE)
            {
              keysInput[i] = keys[i];
              valuesInput[i] = values[i];
            }
          }
        }
        // Sort bucket in shared memory and copy back to global memory.
        else if (size < QUICKSORT_SIZE)
        {
          if (threadIdx.x == 0)
          {
            wasFlipped = flip;
            qsStart[0] = from;
            qsEnd[0] = to;
            qsFlip[0] = flip;
            qsStack = 0;
          }
          __syncthreads();

          while (qsStack >= 0)
          {
            // Get the next sequence.
            if (threadIdx.x == 0)
            {
              from = qsStart[qsStack];
              to = qsEnd[qsStack];
              size = to - from;

              if (!qsFlip[qsStack])
              {
                keys = keysInput;
                keys2 = keysBuffer;
                values = valuesInput;
                values2 = valuesBuffer;
              }
              else
              {
                keys = keysBuffer;
                keys2 = keysInput;
                values = valuesBuffer;
                values2 = valuesInput;
              }
            }	

            __syncthreads();

            // If the sequence is small enough it is sorted in shared memory.
            if (size <= LOCAL_SORT_SIZE)
            {
              if (size > 0)
              {
                unsigned int coal = (from)&0xf;
                for (int i = threadIdx.x; i < size + coal; i += CTA_SIZE)
                {
                  KeyType key = keys[i + from - coal];
                  ValueType value = values[i + from - coal];
                  if (i >= coal)
                  {
                    shared[i - coal] = key;
                    sharedValues[i - coal] = value;
                  }
                }

                __syncthreads();

              // Depending on the size of the sequence to sort, choose a fitting permutation of odd-even-merge-sort.
              // This case-distinction saves a few expensive instructions at run time. For sort sizes > 2048 more case 
              // distinctions would have to be added. Also a minimum block size of 256 is assumed here.
              if (size > 1024) 
              {
                if (CTA_SIZE > 1024) oddEvenSortKeyValueSmall<KeyType, ValueType, StrictWeakOrdering, 2048>(shared, sharedValues, size, comp);
                else oddEvenSortKeyValue<KeyType, ValueType, StrictWeakOrdering, 2048, CTA_SIZE>(shared, sharedValues, size, comp);
              }
              else if (size > 512) 
              {
                if (CTA_SIZE > 512) oddEvenSortKeyValueSmall<KeyType, ValueType, StrictWeakOrdering, 1024>(shared, sharedValues, size, comp);
                else oddEvenSortKeyValue<KeyType, ValueType, StrictWeakOrdering, 1024, CTA_SIZE>(shared, sharedValues, size, comp);
              }
              else if (size > 256)
              {
                if (CTA_SIZE > 256) oddEvenSortKeyValueSmall<KeyType, ValueType, StrictWeakOrdering, 512>(shared, sharedValues, size, comp);
               else oddEvenSortKeyValue<KeyType, ValueType, StrictWeakOrdering, 512, CTA_SIZE>(shared, sharedValues, size, comp);
              }
              else if (size > 128)
              {
                if (CTA_SIZE > 128) oddEvenSortKeyValueSmall<KeyType, ValueType, StrictWeakOrdering, 256>(shared, sharedValues, size, comp);
                else oddEvenSortKeyValue<KeyType, ValueType, StrictWeakOrdering, 256, CTA_SIZE>(shared, sharedValues, size, comp);
              }
              else if (size > 64) 
              {
                if (CTA_SIZE > 64) oddEvenSortKeyValueSmall<KeyType, ValueType, StrictWeakOrdering, 128>(shared, sharedValues, size, comp);
               else oddEvenSortKeyValue<KeyType, ValueType, StrictWeakOrdering, 128, CTA_SIZE>(shared, sharedValues, size, comp);
              }
              else if (size > 32) 
              {
                if (CTA_SIZE > 32) oddEvenSortKeyValueSmall<KeyType, ValueType, StrictWeakOrdering, 64>(shared, sharedValues, size, comp);
                else oddEvenSortKeyValue<KeyType, ValueType, StrictWeakOrdering, 64, CTA_SIZE>(shared, sharedValues, size, comp);
              }
              else oddEvenSortKeyValueSmall<KeyType, ValueType, StrictWeakOrdering, 32>(shared, sharedValues, size, comp);

              for (int i = threadIdx.x; i < size; i += CTA_SIZE)
              {
                keysInput[i + from] = shared[i];
                valuesInput[i + from] = sharedValues[i];
              }
            }

              if (threadIdx.x == 0) 
                --qsStack;

              __syncthreads();
            }
            else
            {
              if (threadIdx.x == 0)
              {
                KeyType mip = SampleSort::min<KeyType>(SampleSort::min<KeyType>(keys[from], keys[to - 1], comp), keys[(from + to) / 2], comp);
                KeyType map = SampleSort::max<KeyType>(SampleSort::max<KeyType>(keys[from], keys[to - 1], comp), keys[(from + to) / 2], comp);
                pivot = SampleSort::min<KeyType>(SampleSort::max<KeyType>(mip / 2 + map / 2, mip, comp), map, comp);
              }

              unsigned int ll = 0;
              unsigned int lm = 0;
              unsigned int lr = 0;

              unsigned int coal = (from)&0xf;

              __syncthreads();

              if (threadIdx.x + from - coal < to)
              {
                KeyType d = keys[threadIdx.x + from - coal];

                if(threadIdx.x >= coal)
                {
                  if      (comp(d, pivot)) ++ll;
                  else if (comp(pivot, d)) ++lr;
                  else                     ++lm;
                }
              }

              for (int i = from + threadIdx.x + CTA_SIZE - coal; i < to; i += CTA_SIZE)
              {
                KeyType d = keys[i];

                if      (comp(d, pivot)) ++ll;
                else if (comp(pivot, d)) ++lr;
                else                     ++lm;
              }

              // Store the result in shared memory for the prefix sums.
              smallBlock[threadIdx.x] = ll;
              equalBlock[threadIdx.x] = lm;
              largeBlock[threadIdx.x] = lr;

              __syncthreads();

              // Blelloch's scan is not optimal, but it is in-place.
              scan3<unsigned int, CTA_SIZE>(smallBlock, equalBlock, largeBlock);

              if (threadIdx.x == 0)
              {
                // Refill the qsStack.
                qsFlip[qsStack+1] = !qsFlip[qsStack];
                qsFlip[qsStack] = !qsFlip[qsStack];

                if (smallBlock[CTA_SIZE] < largeBlock[CTA_SIZE])
                {
                  qsStart[qsStack + 1] = qsStart[qsStack];
                  qsStart[qsStack] = to - largeBlock[CTA_SIZE];
                  qsEnd[qsStack + 1] = from + smallBlock[CTA_SIZE];
                }
                else
                {
                  qsEnd[qsStack + 1] = qsEnd[qsStack];
                  qsEnd[qsStack] = from + smallBlock[CTA_SIZE];
                  qsStart[qsStack + 1] = to - largeBlock[CTA_SIZE];
                }

                ++qsStack;
              }

              __syncthreads();

              unsigned int x = from + smallBlock[threadIdx.x + 1] - 1;
              unsigned int y = to - largeBlock[threadIdx.x + 1];
              unsigned int z = from + smallBlock[CTA_SIZE] + equalBlock[threadIdx.x];

              // Use the offsets generated by the prefix sums to write the elements to their new positions.
              if (threadIdx.x + from - coal < to)
              {
                KeyType d = keys[threadIdx.x + from - coal];
                ValueType vd = values[threadIdx.x + from - coal];

                if (threadIdx.x >= coal)
                {
                  if (comp(d, pivot))
                  {
                    keys2[x] = d;
                    values2[x] = vd;
                    --x;
                  }
                  else if (comp(pivot, d))
                  {
                    keys2[y] = d;
                    values2[y] = vd;
                    ++y;
                  }
                  else
                  {
                    keys2[z] = d;
                    values2[z] = vd;
                    ++z;
                  }
               }
             }

            for (int i = from + threadIdx.x + CTA_SIZE - coal; i < to; i += CTA_SIZE)
            {	
              KeyType d = keys[i];
              ValueType vd = values[i];
 
              if (comp(d, pivot))
              {
                keys2[x] = d;
                values2[x] = vd;
                --x;
              }
              else if (comp(pivot, d))
              {
                keys2[y] = d;
                values2[y] = vd;
                ++y;
              }		
              else
              {
                keys2[z] = d;
                values2[z] = vd;
                ++z;
              }				
            }
            __syncthreads();

            // Write the keys/values equal to the pivot to the output array since their final position is known.
            for (int i = from + smallBlock[CTA_SIZE] + threadIdx.x; i < to - largeBlock[CTA_SIZE]; i += CTA_SIZE)
            {
              keysInput[i] = keys2[i];
              valuesInput[i] = values2[i];
            }
            __syncthreads();
            }
          }

          if (threadIdx.x == 0)
          {
            flip = wasFlipped;
            
          if (!flip)
          {
            keys = keysInput;
            keys2 = keysBuffer;
            values = valuesInput;
            values2 = valuesBuffer;
          }
          else
          {
            keys = keysBuffer;
            keys2 = keysInput;
            values = valuesBuffer;
            values2 = valuesInput;
         }
          }

        }
        // The bucket is too large, put it on the stack for k-way splitting.
        else
        {
          if (threadIdx.x == 0)
          {
            ++largeStack;
            start[largeStack] = from;
            end[largeStack] = to;
            largeBucketsFlip[largeStack] = flip;      
          }
        }

        if (threadIdx.x == 0)
          --newStack;

        __syncthreads();
      }

      __syncthreads();

      if (largeStack >= 0)
      {
        // Pop a large bucket from the stack, taking into account in which of the two keys buffers it resides.
        __syncthreads();
        if (threadIdx.x == 0)
        {
          from = start[largeStack];
          to = end[largeStack];
          size = to - from;
          flip = largeBucketsFlip[largeStack];
          if (!flip)
          {
            keys = keysInput;
            keys2 = keysBuffer;
            values = valuesInput;
            values2 = valuesBuffer;
          }
          else
          {
            keys = keysBuffer;
            keys2 = keysInput;
            values = valuesBuffer;
            values2 = valuesInput;
         }
          --largeStack;
        }
        __syncthreads();

        // K-way split the smallest largest bucket.
        KeyType *sample = &shared[0];	
        for (int i = threadIdx.x; i < (A * K); i += CTA_SIZE)
        {
          int randPos = from + (lrand48NextInt(rng) % size);
          sample[i] = keys[randPos];
        }	
        __syncthreads();

        oddEvenSortSmall<KeyType, StrictWeakOrdering, A_P2 * K>(sample, A * K, comp);

        // Check if all splitters are equal, in which case no bst traversal will be done.
        if (threadIdx.x == 0)
        {
          lSplit = sample[A];
          KeyType hSplit = sample[A * (K - 1)];
          splitConstant = !comp(lSplit, hSplit) && !comp(hSplit, lSplit);
        }
        __syncthreads();

        if (!splitConstant)
        {
          // Create bst and store it in shared memory.
          for (int i = threadIdx.x; i < K; i += CTA_SIZE)
          {
            int l = (int)__log2f(i + 1);
            int pos = (A * K * (1 + 2 * ((i + 1) % (1 << l)))) / (2 << l);
            bst[i] = sample[pos];
          }	
        }

        __syncthreads();

        // Initialize the bucket counters.
        int *bucketCounters = (int*)&shared[0];
        for (int i = threadIdx.x; i < NUM_SCATTER * K; i += CTA_SIZE)
          bucketCounters[i] = 0;

        __syncthreads();

        // Each thread loads 1 element and traverses the search tree, until
        // all elements are done. This is done in batches so the bucket
        // positions can be kept in shared memory.

        if (!splitConstant)
        {
          for (int i = from + threadIdx.x; i < to; i += CTA_SIZE)
          { 
              KeyType d = keys[i];		
              int bucketPos = 1;

              // Traverse bst.
              for (int j = 0; j < lg<K>::result; ++j)
              {
                if (comp(bst[bucketPos - 1], d)) bucketPos = (bucketPos << 1) + 1;
                else                             bucketPos <<= 1;
              }

              bucketPos = bucketPos - K;
              atomicAdd(bucketCounters + bucketPos * NUM_SCATTER + threadIdx.x % NUM_SCATTER, 1);
          }
        }
        else
        {
          for (int i = from  + threadIdx.x; i < to; i += CTA_SIZE)
          {     
              KeyType d = keys[i];		
              int bucketPos = 1;

              if      (comp(lSplit, d)) bucketPos = 2;
              else if (comp(d, lSplit)) bucketPos = 0;    

              atomicAdd(bucketCounters + bucketPos * NUM_SCATTER + threadIdx.x % NUM_SCATTER, 1);
          }
        }

        __syncthreads();

        brent_kung_exclusive<int, K * NUM_SCATTER>(bucketCounters);

        // Copy the new bucket positions. If all splitters were equal, only
        // the first three bucket can be non-empty and have to go onto the stack.
        if (!splitConstant)
        {
          for (int i = threadIdx.x; i < K; i += CTA_SIZE)
            newBucketsStart[i] = from + bucketCounters[i * NUM_SCATTER];
        }
        else
        {
          for (int i = threadIdx.x; i < 3; i += CTA_SIZE)
            newBucketsStart[i] = from + bucketCounters[i * NUM_SCATTER];
        }

        // Find the bucket positions in batches of block size, now writing 
        // out elements each time.
        if (!splitConstant)
        {
          for (int i = from + threadIdx.x; i < to; i += CTA_SIZE)
          {       
              KeyType d = keys[i];
              ValueType vd = values[i];
              int bucketPos = 1;

              // Traverse bst.
              for (int j = 0; j < lg<K>::result; ++j)
              {
                if (comp(bst[bucketPos - 1], d)) bucketPos = (bucketPos << 1) + 1;
                else                             bucketPos <<= 1;
              }		

              bucketPos = bucketPos - K;
              int outputPos = from + atomicAdd(bucketCounters + bucketPos * NUM_SCATTER + threadIdx.x % NUM_SCATTER, 1);
              keys2[outputPos] = d;
              values2[outputPos] = vd;
          }
        }
        else
        {
          for (int i = from + threadIdx.x; i < to; i += CTA_SIZE)
          {      
              KeyType d = keys[i];
              ValueType vd = values[i];
              int bucketPos = 1;

              if      (comp(lSplit, d)) bucketPos = 2;
              else if (comp(d, lSplit)) bucketPos = 0;

              int outputPos = from + atomicAdd(bucketCounters + bucketPos * NUM_SCATTER + threadIdx.x % NUM_SCATTER, 1);
              keys2[outputPos] = d;
              values2[outputPos] = vd;
          }
        }

        __syncthreads();

        // Prepare for the next iteration.
        if (threadIdx.x == 0)
        {          
          if (flip)
          {
            keys = keysInput;
            keys2 = keysBuffer;
            values = valuesInput;
            values2 = valuesBuffer;
          }
          else
          {
            keys = keysBuffer;
            keys2 = keysInput;
            values = valuesBuffer;
            values2 = valuesInput;
         }
          flip = !flip;

          // If all splitters were equal only the first 3 buckets have to be put onto the stack.
          if (!splitConstant)
          {
            newBucketsStart[K] = to;
            newStack = K - 1;
          }
          else
          {
            newBucketsStart[3] = to;
            newStack = 2;
          }
        }
        __syncthreads();
      }	
    }
    lrand48StoreStateSimple(rng); 
  }
}