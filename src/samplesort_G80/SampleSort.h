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

#ifndef __SAMPLE_SORT__
#define __SAMPLE_SORT__

#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/type_traits.h>
#include <thrust/functional.h>
#include "samplesort/detail/Lrand48.inl"
#include "samplesort/detail/CTAScan.inl"
#include "samplesort/detail/CTAOddEven.inl"

namespace SampleSort
{  
  // Compute the log2 of an integer at compile time.
  template <unsigned int N, unsigned int Cur = 0>
  struct lg
  {
    static const unsigned int result = lg<(N >> 1),Cur+1>::result;
  };

  template <unsigned int Cur>
  struct lg<1,Cur>
  {
    static const unsigned int result = Cur;
  };

  template<typename T, typename BinaryPredicate>
  __device__ T min(const T &lhs, const T &rhs, BinaryPredicate comp)
  {
    return comp(lhs, rhs) ? lhs : rhs;
  }

  template<typename T>
  __device__ T min(const T &lhs, const T &rhs)
  {
    return lhs < rhs ? lhs : rhs;
  }

  template<typename T, typename BinaryPredicate>
  __device__  T max(const T &lhs, const T &rhs, BinaryPredicate comp)
  {
    return comp(lhs,rhs) ? rhs : lhs;
  }

  template<typename T>
  __device__  T max(const T &lhs, const T &rhs)
  {
    return lhs < rhs ? rhs : lhs;
  }

  const int K = 128;
  __constant__ unsigned long long bstCache[K];

  template<typename RandomAccessIterator1, typename RandomAccessIterator2, typename StrictWeakOrdering, 
    unsigned int A, unsigned int LOCAL_SORT_SIZE, unsigned int LOCAL_SORT_SIZE_KV>
    void sort(RandomAccessIterator1 keysBegin, RandomAccessIterator1 keysEnd, 
    RandomAccessIterator2 valuesBegin, StrictWeakOrdering comp, bool keysOnly, int numCTAs = 350);

  void sort_by_key(unsigned int *keys, unsigned int *keysEnd, unsigned int *values);
  void sort_by_key(unsigned long long *keys, unsigned long long *keysEnd, unsigned int *values);
  void sort(unsigned int *keys, unsigned int *keysEnd);
  void sort(unsigned long long *keys, unsigned long long *keysEnd);
}

#endif
