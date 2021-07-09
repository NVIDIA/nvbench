# 3.18.3 is needed for a  MSVC + NVCC + C++17 bugfix.
cmake_minimum_required(VERSION 3.18.3)

# CXX to work around issues with CUDA-only CMake projects.
project(NVBench CUDA CXX)

option(NVBench_ENABLE_TESTING "Build NVBench testing suite." OFF)
option(NVBench_ENABLE_EXAMPLES "Build NVBench examples." OFF)

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

# Following recipe from
# http://github.com/cpm-cmake/CPM.cmake/blob/master/examples/json/CMakeLists.txt
# Download the zips because the repo takes an excessively long time to clone.
CPMAddPackage(
  NAME nlohmann_json

  # I'm waiting for https://github.com/nlohmann/json/issues/2676 to be fixed,
  # leave this in to simplify testing patches as they come out. Update the
  # `nvbench_json` target too when switching branches.
  # Development version:
#  VERSION develop
#  URL https://github.com/nlohmann/json/archive/refs/heads/develop.zip
#  OPTIONS JSON_MultipleHeaders ON

  # Latest release headers:
  VERSION 3.9.1
  URL https://github.com/nlohmann/json/releases/download/v3.9.1/include.zip
  URL_HASH SHA256=6bea5877b1541d353bd77bdfbdb2696333ae5ed8f9e8cc22df657192218cad91
  PATCH_COMMAND
    # Work around compiler bug in nvcc 11.0, see NVIDIA/NVBench#18
    ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/nlohmann_json.hpp" "./include/nlohmann/json.hpp"
)

# Development branch:
#add_library(nvbench_json INTERFACE)
#target_link_libraries(nvbench_json INTERFACE nlohmann_json)

# Release headers
add_library(nvbench_json INTERFACE)
target_include_directories(nvbench_json SYSTEM INTERFACE
  "${nlohmann_json_SOURCE_DIR}/include"
)

# Builds all NVBench targets (libs, tests, examples, etc).
add_custom_target(nvbench.all)

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
