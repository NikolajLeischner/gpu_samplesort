cmake_minimum_required(VERSION 2.6.2)

project (benchmark)

set(CMAKE_MODULE_PATH 
  ${benchmark_SOURCE_DIR}/CMake/
  ${benchmark_SOURCE_DIR}/CMake/Packages/
  ${benchmark_SOURCE_DIR}/CMake/Utils/
)
if (CMAKE_BUILD_TYPE)
  string(TOLOWER ${CMAKE_BUILD_TYPE} benchmark_BUILD_TYPE)
else ()
  set(benchmark_BUILD_TYPE "")
endif ()
include(${benchmark_SOURCE_DIR}/CMake/Utils/AutoGroups.cmake)

add_definitions(-DNOMINMAX)

find_package(CUDA REQUIRED)

include_directories(
  ${benchmark_SOURCE_DIR}
  ${benchmark_SOURCE_DIR}/benchmark
  ${benchmark_SOURCE_DIR}/samplesort
)

add_subdirectory(benchmark)