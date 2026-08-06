cmake_minimum_required(VERSION 3.30.4)
project(NVBenchExternalFmt CUDA CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED FMT_SOURCE_DIR)
  message(FATAL_ERROR "FMT_SOURCE_DIR must be set.")
endif()

if(NOT DEFINED NVBench_SOURCE_DIR)
  message(FATAL_ERROR "NVBench_SOURCE_DIR must be set.")
endif()

set(external_fmt_config_dir "${CMAKE_CURRENT_BINARY_DIR}/external_fmt/lib/cmake/fmt")
file(MAKE_DIRECTORY "${external_fmt_config_dir}")
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/fmt-config.cmake.in"
  "${external_fmt_config_dir}/fmt-config.cmake"
  @ONLY
)
set(fmt_DIR "${external_fmt_config_dir}" CACHE PATH "" FORCE)
find_package(fmt REQUIRED CONFIG)

if(NOT TARGET fmt::fmt)
  message(FATAL_ERROR "External fmt did not provide fmt::fmt.")
endif()

set(NVBench_ENABLE_CUPTI OFF CACHE BOOL "" FORCE)
set(NVBench_ENABLE_EXECUTABLES OFF CACHE BOOL "" FORCE)
set(NVBench_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(NVBench_ENABLE_HEADER_TESTING OFF CACHE BOOL "" FORCE)
set(NVBench_ENABLE_INSTALL_RULES OFF CACHE BOOL "" FORCE)
set(NVBench_ENABLE_NVML OFF CACHE BOOL "" FORCE)
set(NVBench_ENABLE_TESTING OFF CACHE BOOL "" FORCE)

add_subdirectory("${NVBench_SOURCE_DIR}" nvbench)

if(NOT TARGET nvbench::nvbench)
  message(FATAL_ERROR "NVBench did not define nvbench::nvbench.")
endif()
