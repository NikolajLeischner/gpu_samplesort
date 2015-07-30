cmake_minimum_required(VERSION 2.8)

project (benchmark)

find_package(CUDA QUIET REQUIRED)

include_directories(
  ${benchmark_SOURCE_DIR}/src
  ${benchmark_SOURCE_DIR}/test
  ${benchmark_SOURCE_DIR}/test/bandit
)

add_definitions(-DNOMINMAX)
add_subdirectory(src)
add_subdirectory(test)

enable_testing()
add_test (NAME run_tests COMMAND tests)