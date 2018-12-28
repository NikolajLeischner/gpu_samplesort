include_directories(
  benchmark
  samplesort
  external)

SET(SOURCE_FILES 
  benchmark/distributions.h
  benchmark/distributions.cpp
  benchmark/dispatching.h
  benchmark/dispatching.cpp
  benchmark/gpu_dispatching.h
  benchmark/keytype.h
  benchmark/keytype.cpp
  benchmark/output.h
  benchmark/output.cpp
  benchmark/result.h
  benchmark/settings.h
  benchmark/settings.cpp
  benchmark/timer.h
  benchmark/timer.cpp
  samplesort/samplesort.h
  samplesort/samplesort.cu
  benchmark/gpu_dispatching.cu
)

ADD_LIBRARY(gpu_samplesort ${SOURCE_FILES})
ADD_EXECUTABLE(benchmark benchmark.cpp)
TARGET_LINK_LIBRARIES(benchmark gpu_samplesort)
