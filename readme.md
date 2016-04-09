This is a revised implementation of GPU Sample Sort. The code is intended to be more readable than the original implementation. Scripts for benchmarking the algorithm, and for generating charts are included.

Building

Generating Charts



Use CMake (>= 2.8) to generate a makefile or a Visual Studio project.

The code has been tested with CUDA 3.0 on G80 and G100 (Fermi) cards on 64bit Windows 7 and 64bit Linux, and with Cuda 5.0 on a Geforce GTX 560. The code requires sm_13 to compile (you must tell CMake to use the compile flag -arch=sm_13 or -arch=sm_20 for nvcc). To use the version without shared memory atomics copy the contents from src/samplesort_G80 to src/samplesort.

Parameters are tuned for G100 and G80 respectively, G200 may like other parameters better (e.g. a smaller local sort size to lower shared memory usage). 