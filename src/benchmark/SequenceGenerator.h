#ifndef Sequence_Generator
#define Sequence_Generator

#include "../samplesort/detail/MersenneTwister.h"

namespace GpuSortingBenchmark
{

namespace SequenceGenerator
{
	/* Fills an array with a distribution of random values.The distributions described in http://www.umiacs.umd.edu/research/EXPAR/papers/3669/node5.html#SECTION00041000000000000000,
	from 0 to 9: ZERO, UNIFORM, GAUSSIAN, BUCKET, STAGGERED, SORTED-ASCENDING, G-GROUPS, RANDOMIZED DUPLICATES, DETERMINISTIC DUPLICATES, SORTED-DESCENDING. */	
        template <typename element>
	void fill(MtRng32 *rng32, MtRng64 *rng64, element* data, unsigned int size, int distType, unsigned int numBits, unsigned int numSamples, 
          unsigned int P, unsigned int G, element RANGE, bool use64bits = false);
}

}

#include "SequenceGenerator.inl"

#endif
