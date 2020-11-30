This is an updated implementation of GPU Sample Sort. The algorihm is the same as in the publication, but the code has been cleaned up & modified to work with more recent versions of CUDA and C++ compilers. Scripts for benchmarking the algorithm and for generating charts are included as well.

The code has been compiled & tested with CUDA 11 + MSVC 2019 16.8 on Windows 10 and CUDA 9 + GCC 6.5 on Ubuntu 18.10.

# Requirements

* CUDA 9+
* A compatible C++ compiler (e.g. GCC, MSVC)
* CMAKE 3.12
* Python 3.6 for generating charts

# References

* https://arxiv.org/abs/0909.5649
* https://en.wikipedia.org/wiki/Samplesort
