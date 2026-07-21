cmake_minimum_required(VERSION 3.30...4.0)

# CUDA is transitive dependency of nvbench
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX CUDA)

find_package(Python REQUIRED COMPONENTS Development.Module)
find_package(CUDAToolkit REQUIRED)

# Get CMake package manager
set(_cpm_download_location ${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake)
file(
  DOWNLOAD
  https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.42.0/CPM.cmake
  ${_cpm_download_location}
  EXPECTED_HASH SHA256=2020b4fc42dba44817983e06342e682ecfc3d2f484a581f11cc5731fbe4dce8a
)
include(${_cpm_download_location})

# Determine CUDA major version for directory structure
set(CUDA_VERSION_DIR "cu${CUDAToolkit_VERSION_MAJOR}")
set(NVBENCH_INSTALL_DIR      "cuda/bench/${CUDA_VERSION_DIR}/nvbench")
set(CMAKE_INSTALL_INCLUDEDIR "${NVBENCH_INSTALL_DIR}/include")
set(CMAKE_INSTALL_LIBDIR     "${NVBENCH_INSTALL_DIR}/lib")
CPMAddPackage(
  NAME nvbench
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..
  OPTIONS
    "NVBench_ENABLE_TESTING OFF"
    "NVBench_ENABLE_INSTALL_RULES ON"
    "NVBench_ENABLE_CUPTI OFF"
    "NVBench_ENABLE_EXECUTABLES OFF"
    "NVBench_ENABLE_EXAMPLES OFF"
  FIND_PACKAGE_ARGS CONFIG REQUIRED
)

CPMAddPackage("gh:pybind/pybind11@3.0.4")

message(STATUS "Building extension for CUDA ${CUDAToolkit_VERSION_MAJOR}, output directory: cuda/bench/${CUDA_VERSION_DIR}")

add_library(_nvbench MODULE src/py_nvbench.cpp)
target_include_directories(_nvbench PRIVATE ${Python_INCLUDE_DIRS})
target_link_libraries(_nvbench PUBLIC nvbench::nvbench)
target_link_libraries(_nvbench PRIVATE CUDA::cudart_static CUDA::cuda_driver pybind11::headers)

set_target_properties(_nvbench PROPERTIES INSTALL_RPATH "$ORIGIN/nvbench/lib")
set_target_properties(_nvbench PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)
set_target_properties(_nvbench PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(_nvbench PROPERTIES PREFIX "" SUFFIX "${PYTHON_MODULE_EXTENSION}")

install(TARGETS _nvbench DESTINATION cuda/bench/${CUDA_VERSION_DIR})
