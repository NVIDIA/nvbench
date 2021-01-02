cmake_minimum_required(VERSION 3.18)

project(nvbench CUDA)

# Setup some vars for CPM packages:
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include(CPM)

CPMAddPackage(
  NAME benchmark
  GITHUB_REPOSITORY google/benchmark
  VERSION 1.5.2
  OPTIONS
  "BENCHMARK_ENABLE_TESTING Off"
  # Note from cucollections:
  # The REGEX feature test fails when gbench's cmake is run under CPM w/ gcc5.4 because it doesn't assume C++11
  # Additionally, attempting to set the CMAKE_CXX_VERSION here doesn't propogate to the feature test build
  # Therefore, we just disable the feature test and assume platforms we care about have a regex impl available
  "RUN_HAVE_STD_REGEX 0"
)

CPMAddPackage(
  NAME fmt
  GITHUB_REPOSITORY fmtlib/fmt
  GIT_TAG 7.1.3
)

# Why does CMake make it so difficult to enable cuda C++17 with interface
# properties...Resorting to brute force to unblock.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17")

add_subdirectory(nvbench)
add_subdirectory(testing)
