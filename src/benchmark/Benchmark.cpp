#include <algorithm>
#include <iostream>
#include <math.h>
#include "SampleSortWrapper.h"
#include "samplesort/detail/MersenneTwister.h"
#include "SequenceGenerator.h"

// Minimalistic benchmark, showcasing the usage of GPU Sample Sort. 

// Compare with std::sort to check for correctness. 
bool isSorted(unsigned int *data, unsigned int *sorted, unsigned int size)
{
  unsigned int *dataCopy = new unsigned int[size];
  bool isSorted = true;

  std::copy(data, data + size, dataCopy);
  std::sort(dataCopy, dataCopy + size);
  int numError = 0;

  for (int i = 0; i < size; ++i) 
  {
    if (sorted[i] != dataCopy[i]) 
    {      
      numError++;
      isSorted = false;
    }
  }

  if (!isSorted)  
    std::cout << std::endl << "Found " << numError << " errors." << std::endl;

  delete [] dataCopy;
  return isSorted;	
}

/* Benchmark GPU Sample Sort.
 * param dist: input distribution.
 * param P, G, RANGE, numBits, numSamples: distribution parameters.
 * param size: input size.
 * param numSeq: number of inputs to test.
 * param checkResults: check results for correctness.
 */
const int REPETITION_CONST = 262144;
const std::string distNames[] = { "constant", "uniform", "gaussian", "bucket", "staggered", "sorted-ascending",\
"g-group", "randomized-duplicates", "deterministic-duplicates", "sorted-descending" };

void runBenchmark(int dist, int P, int G, unsigned int RANGE, int numBits, int numSamples, 
                  unsigned int size, int numSeq, bool checkResults=true)
{
  using namespace GpuSortingBenchmark;

  MtRng32 *rng32 = new MtRng32(5489); 
  MtRng64 *rng64 = new MtRng64(5489); 

  const unsigned int numRepeat = std::max((unsigned int)1, REPETITION_CONST / size);

  unsigned int *data = new unsigned int[size]; 
  std::fill(data, data + size, 0);
  unsigned int *dataCopy = new unsigned int[size]; 
  std::fill(dataCopy, dataCopy + size, 0);

  for (int i = 0; i < numSeq; ++i)
  {
    SequenceGenerator::fill(rng32, rng64, data, size, dist, numBits, numSamples, P, G, RANGE);

    double time = 0;
    double singleTime = 0;
    double transferTime = 0;
    double singleTransferTime = 0;
    bool valid = true;

    benchSampleSort(dataCopy, size, &singleTime, &singleTransferTime);

    for (unsigned int n = 0; n < numRepeat; ++n)
    {
      std::copy(data, data+size, dataCopy);
      benchSampleSort(dataCopy, size, &singleTime, &singleTransferTime);
      time += singleTime;
      transferTime += singleTransferTime;
      if (checkResults) { if (!isSorted(data, dataCopy, size)) valid = false; }				
    }

    time /= numRepeat;
    transferTime /= numRepeat;

    std::cout << "Time: " << (float)time << " Transfer time: " << (float)transferTime << " Size: " 
      << size << " Distribution: "  << distNames[dist].c_str() << " Samples: " << numSamples << " Bits: " 
      << numBits << " P: " << P << " G: " << G << " RANGE: " << RANGE << " Valid: " << valid << std::endl ; 
  }
  delete [] data;
  delete [] dataCopy;
  delete rng32;
  delete rng64;
}

int main(int argc, char *argv[]) 
{  
  using namespace GpuSortingBenchmark;

  int dist = 1;
  int numSeq = 3;

  warmUp();

  for (int i = (2 << 15); i < (2 << 25); i *= 2)
    runBenchmark(dist, 128, 8, 32, 32, 1, i, numSeq, true);

  cleanUp();

  return 0;	
}

