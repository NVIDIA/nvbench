# 3.18.3 is needed for a  MSVC + NVCC + C++17 bugfix.
cmake_minimum_required(VERSION 3.18.3)

# CXX to work around issues with CUDA-only CMake projects.
project(NVBench CUDA CXX)

option(NVBench_ENABLE_TESTING "Build NVBench testing suite." OFF)

# Setup some vars for CPM packages:
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/")

# NVBench requires C++17.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# TODO this probably should use GNUInstallDirs or something.
set(NVBench_LIBRARY_OUTPUT_DIR "${CMAKE_BINARY_DIR}/lib")
set(NVBench_EXECUTABLE_OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin")

include(CPM)

CPMAddPackage(
  NAME fmt
  GITHUB_REPOSITORY fmtlib/fmt
  GIT_TAG 7.1.3
)

add_subdirectory(nvbench)

if (NVBench_ENABLE_TESTING)
  enable_testing()
  add_subdirectory(testing)
endif()
