set(SOURCES
  run_tests.cpp
  bandit_test.cpp
)

add_executable(tests ${SOURCES} )
target_link_libraries(tests gpu_samplesort ${CUDA_LIBRARIES})

