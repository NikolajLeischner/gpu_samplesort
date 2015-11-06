include_directories(
  benchmark
  samplesort
  external)

SET(SOURCE_FILES 
  benchmark/distributions.h
  benchmark/distributions.cpp
  benchmark/execution.h
  benchmark/execution.cpp
  benchmark/keytype.h
  benchmark/keytype.cpp
  benchmark/output.h
  benchmark/output.cpp
  benchmark/settings.h
  benchmark/settings.cpp
  benchmark/timer.h
  benchmark/timer.cpp
  samplesort/samplesort.h
)
CUDA_COMPILE(CUDA_FILES 
)

CUDA_ADD_LIBRARY(gpu_samplesort ${CUDA_FILES} ${SOURCE_FILES})
ADD_EXECUTABLE(benchmark benchmark.cpp)
TARGET_LINK_LIBRARIES(benchmark gpu_samplesort ${CUDA_LIBRARIES})

