# 3.18.3 is needed for a  MSVC + NVCC + C++17 bugfix.
cmake_minimum_required(VERSION 3.18.3)

project(nvbench CUDA)

# Setup some vars for CPM packages:
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include(CPM)

CPMAddPackage(
  NAME fmt
  GITHUB_REPOSITORY fmtlib/fmt
  GIT_TAG 7.1.3
)

add_subdirectory(nvbench)
add_subdirectory(testing)
