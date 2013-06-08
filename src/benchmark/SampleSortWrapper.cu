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

#include "SampleSortWrapper.h"
#include "Timer.h"
#include "samplesort/SampleSort.h"

namespace GpuSortingBenchmark
{
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

  // Copies data from the main memory to the GPU, sorts it, copies it back. Measures the
  // running time and data transfer time.
  void benchSampleSort(unsigned int *keys, unsigned int size, double* time, double* transferTime)
  {
    Timer timer;
    unsigned int *dKeys = 0;
    size_t memSize = size * sizeof(unsigned int);
    timer.start();
    cudaMalloc((void**) &dKeys, memSize);
    cudaMemcpy(dKeys, keys, memSize, cudaMemcpyHostToDevice);
    *transferTime = timer.stop();

    cudaThreadSynchronize();
    timer.start();
    SampleSort::sort(dKeys, dKeys + size);
    cudaThreadSynchronize();
    *time = timer.stop();

    timer.start();	
    cudaMemcpy(keys, dKeys, memSize, cudaMemcpyDeviceToHost);  
    cudaFree(dKeys);
    *transferTime += timer.stop();
  }
}


