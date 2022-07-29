This is an updated implementation of GPU Sample Sort. The algorithm is the same as in the publication, but the code has been cleaned up & modified to work with more recent versions of CUDA and C++ compilers. Scripts for benchmarking the algorithm and for generating charts are included as well.

The code has been compiled & tested with CUDA 11.6 + MSVC 2022 17.0 on Windows 11.

# Requirements

* CUDA 11+
* A compatible C++ compiler (e.g. GCC, MSVC)
* CMAKE 3.18
* Python 3.10 for generating charts

# References

* https://arxiv.org/abs/0909.5649
* https://en.wikipedia.org/wiki/Samplesort
