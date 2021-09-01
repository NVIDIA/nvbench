cmake_minimum_required(VERSION 3.20.1)
project(NVBenchTestExport CUDA CXX)

message(STATUS "NVBench_DIR=${NVBench_DIR}")
find_package(NVBench)

add_executable(test_bench test_bench.cu)
target_link_libraries(test_bench PRIVATE nvbench::main)
enable_testing()
add_test(NAME test_bench COMMAND "$<TARGET_FILE:test_bench>" --timeout 1)

# Need to add the CUPTI path to LD_LIBRARY_PATH to make sure CUPTI libraries
# are found at runtime:
if (UNIX AND TARGET nvbench::cupti)
  get_property(cupti_lib_path TARGET nvbench::cupti PROPERTY IMPORTED_LOCATION)
  cmake_path(GET cupti_lib_path PARENT_PATH cupti_lib_path)
  set_property(TEST test_bench PROPERTY ENVIRONMENT "LD_LIBRARY_PATH=${cupti_lib_path}")
endif()
