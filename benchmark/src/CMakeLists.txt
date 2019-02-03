CMAKE_MINIMUM_REQUIRED(VERSION 3.12)

PROJECT(sort_benchmark)

ENABLE_LANGUAGE(CUDA)

SET(CMAKE_CUDA_STANDARD 11)
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
SET(CMAKE_CXX_STANDARD_REQUIRED TRUE)
SET(CMAKE_CXX_STANDARD_REQUIRED TRUE)

IF (MSVC)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
ENDIF (MSVC)
IF (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -march=native")
ENDIF ()

INCLUDE_DIRECTORIES(
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
        benchmark/constants.cpp
        benchmark/timer.h
        benchmark/timer.cpp
        samplesort/samplesort.cu
        samplesort/samplesort.h
        benchmark/gpu_dispatching.cu
        benchmark/parallel_sort.h
        )

ADD_LIBRARY(gpu_samplesort ${SOURCE_FILES})

ADD_EXECUTABLE(${PROJECT_NAME} benchmark/benchmark.cpp)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} gpu_samplesort ${CUDA_LIBRARIES})
