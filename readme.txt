Use CMake to generate a makefile or a Visual Studio project.

The Thrust library is required (http://code.google.com/p/thrust/) to run GPU Sample Sort. For your convenience it is included here. 

The code has been tested with CUDA 3.0 on G80 and G100 (Fermi) cards on 64bit Windows 7 and 64bit Linux. The code requires sm_13 to compile (you must tell CMake to use the compile flag -arch=sm_13 or -arch=sm_20 for nvcc). To use the version without shared memory atomics copy the contents from src/samplesort_G80 to src/samplesort.

Parameters are tuned for G100 and G80 respectively, G200 may like other parameters better (e.g. a smaller local sort size to lower shared memory usage). 