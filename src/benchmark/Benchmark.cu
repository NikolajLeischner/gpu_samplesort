#include <algorithm>
#include <iostream>
#include <math.h>
#include "SampleSortWrapper.h"
#include "samplesort/detail/MersenneTwister.h"
#include "SequenceGenerator.h"

// Minimalistic benchmark, showcasing the usage of GPU Sample Sort. Just include
// SampleSort.h and call SampleSort::sort..


// Call this for safety, CUDA may sometimes fail to exit gracefully by itself.
void cleanUp()
{
  cudaThreadExit();
}

// Call this to get rid of the overhead for the first cuda call before measuring anything.
void warmUp()
{
  int *dData = 0;
  cudaMalloc((void**) &dData, sizeof(dData));
  cudaFree(dData); 
}

// Compare with std::sort to check for correctness. 
bool isSorted(unsigned int *data, unsigned int *sorted, unsigned int size)
{
  unsigned int *dataCopy = new unsigned int[size];
  bool isSorted = true;

  std::copy(data, data + size, dataCopy);
  std::sort(dataCopy, dataCopy + size);
  int numError = 0;

  for (int i = 0; i < size; i++) 
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

// Copies data from the main memory to the GPU, sorts it, copies it back. Measures the
// running time and data transfer time.
void sampleSort(unsigned int *keys, unsigned int size, double* time, double* transferTime)
{
  float t = 0.0f;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);  

  unsigned int *dKeys = 0;
  size_t memSize = size * sizeof(unsigned int);
  cudaEventRecord(start, 0);	
  cudaMalloc((void**) &dKeys, memSize);
  cudaMemcpy(dKeys, keys, memSize, cudaMemcpyHostToDevice);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&t, start, stop);	
  *transferTime = t;	

  cudaEventRecord(start, 0);	
  SampleSort::sort(dKeys, size);
  cudaThreadSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&t, start, stop);	
  *time = t;

  cudaEventRecord(start, 0);	
  cudaMemcpy(keys, dKeys, memSize, cudaMemcpyDeviceToHost);  
  cudaFree(dKeys);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&t, start, stop);	

  *transferTime += t;	

  cudaEventDestroy(start);
  cudaEventDestroy(stop);	
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

  for (int i = 0; i < numSeq; i++)
  {
    SequenceGenerator::fill(rng32, rng64, data, size, dist, numBits, numSamples, P, G, RANGE);

    double time = 0;
    double singleTime = 0;
    double transferTime = 0;
    double singleTransferTime = 0;
    bool valid = true;

    sampleSort(dataCopy, size, &singleTime, &singleTransferTime);

    for (unsigned int n = 0; n < numRepeat; n++)
    {
      std::copy(data, data+size, dataCopy);
      sampleSort(dataCopy, size, &singleTime, &singleTransferTime);
      time += singleTime;
      transferTime += singleTransferTime;
      if (checkResults) { if (!isSorted(data, dataCopy, size)) valid = false; }				
    }

    time /= numRepeat;
    transferTime /= numRepeat;

    /*std::cout << "Time: " << (float)time << " Transfer time: " << (float)transferTime << " Size: " 
      << size << " Distribution: " << distNames[dist] << " Samples: " << numSamples << " Bits: " 
      << numBits << " P: " << P << " G: " << G << " RANGE: " << RANGE << " Valid: " << valid << std::endl ; */
  }
  delete [] data;
  delete [] dataCopy;
  delete rng32;
  delete rng64;
}

int main(int argc, char *argv[]) 
{  
  int dist = 1;
  int numSeq = 3;

  warmUp();

  for (int i = (2 << 15); i < (2 << 25); i *= 2)
    runBenchmark(dist, 128, 8, 32, 32, 1, i, numSeq, true);

  cleanUp();

  return 0;	
}

