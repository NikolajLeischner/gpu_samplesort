include_directories(
  benchmark
  samplesort
  tclap)

SET(SOURCE_FILES 
  benchmark/timer.cpp
  benchmark/timer.h
  benchmark/settings.cpp
  benchmark/settings.h
  benchmark/keytype.cpp
  benchmark/keytype.h
  benchmark/distributions.h
  benchmark/distributions.cpp
  samplesort/samplesort.h
)
CUDA_COMPILE(CUDA_FILES 
)

CUDA_ADD_LIBRARY(gpu_samplesort ${CUDA_FILES} ${SOURCE_FILES})
ADD_EXECUTABLE(benchmark benchmark.cpp)
TARGET_LINK_LIBRARIES(benchmark gpu_samplesort ${CUDA_LIBRARIES})

