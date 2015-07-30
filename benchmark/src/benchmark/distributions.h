#pragma once

namespace GpuSortingBenchmark
{
	/* Fills an array with a distribution of random values.The distributions described in http://www.umiacs.umd.edu/research/EXPAR/papers/3669/node5.html#SECTION00041000000000000000,
	from 0 to 9: ZERO, UNIFORM, GAUSSIAN, BUCKET, STAGGERED, SORTED-ASCENDING, G-GROUPS, RANDOMIZED DUPLICATES, DETERMINISTIC DUPLICATES, SORTED-DESCENDING. */	
template<typename T>
abstract class Distribution
{
public:
  const T* const begin() const;
  size_t size() const;
  size_t memory_size() const;
    
  protected:
    Distribution(size_t size);

    T randomValue(size_t numBits, size_t numSamples);
};

}

#include "distributions.inl"
