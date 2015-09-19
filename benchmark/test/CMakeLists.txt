set(SOURCES
  run_tests.cpp
  distributions_test.cpp
)

add_executable(tests ${SOURCES} )
target_link_libraries(tests gpu_samplesort ${CUDA_LIBRARIES})

