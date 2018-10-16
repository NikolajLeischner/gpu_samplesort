cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 14)
SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")


project (benchmark)

find_package(CUDA QUIET REQUIRED)

include_directories(
  ${benchmark_SOURCE_DIR}/src
  ${benchmark_SOURCE_DIR}/test
  ${benchmark_SOURCE_DIR}/test/bandit
)

add_subdirectory(src)
add_subdirectory(test)

enable_testing()
add_test (NAME run_tests COMMAND tests)