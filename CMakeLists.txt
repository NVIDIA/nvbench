# 3.20.1 required for rapids-cmake
cmake_minimum_required(VERSION 3.20.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(cmake/NVBenchRapidsCMake.cmake)
nvbench_load_rapids_cmake()

project(NVBench
  LANGUAGES CUDA CXX # CXX to work around issues with CUDA-only CMake projects.
  VERSION 0.1.0
)

nvbench_init_rapids_cmake()

include(cmake/NVBenchConfigTarget.cmake)
include(cmake/NVBenchDependencies.cmake)
include(cmake/NVBenchExports.cmake)
include(cmake/NVBenchInstallRules.cmake)
include(cmake/NVBenchUtilities.cmake)

message(STATUS "NVBench CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

option(NVBench_ENABLE_TESTING "Build NVBench testing suite." OFF)
option(NVBench_ENABLE_EXAMPLES "Build NVBench examples." OFF)

add_subdirectory(nvbench)

if (NVBench_ENABLE_EXAMPLES OR NVBench_ENABLE_TESTING)
  enable_testing()
endif()

if (NVBench_ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()

if (NVBench_ENABLE_TESTING)
  add_subdirectory(testing)
endif()

nvbench_generate_exports()
