cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")

project (benchmark)

enable_language(CUDA)

include_directories(
  ${benchmark_SOURCE_DIR}/src
  ${benchmark_SOURCE_DIR}/test
  ${benchmark_SOURCE_DIR}/test/bandit
)

add_subdirectory(src)
add_subdirectory(test)

enable_testing()
add_test (NAME run_tests COMMAND tests)
