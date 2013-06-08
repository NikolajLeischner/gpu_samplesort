# Anything other than -D or /D is not passed along to nvcc.

set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
set(source_files 
  SampleSortWrapper.h
  Timer.cpp
  Timer.h
  SequenceGenerator.h
  ../samplesort/SampleSort.h
)
CUDA_COMPILE(CUDA_FILES 
  SampleSortWrapper.cu
  ../samplesort/SampleSort.cu
)

CUDA_ADD_EXECUTABLE(benchmark
  ${CUDA_FILES}
  ${source_files}
  Benchmark.cpp
  )
TARGET_LINK_LIBRARIES(benchmark
  ${CUDA_LIBRARIES}
 )

# Add a special target to clean nvcc generated files.
CUDA_BUILD_CLEAN_TARGET()

