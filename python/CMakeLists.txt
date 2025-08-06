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

CPMAddPackage(
   NAME nvbench
   SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..
   OPTIONS "NVBench_INSTALL_RULES ON"
   FIND_PACKAGE_ARGS CONFIG REQUIRED
)

CPMAddPackage("gh:pybind/pybind11@3.0.0")

pybind11_add_module(_nvbench MODULE src/py_nvbench.cpp)
target_link_libraries(_nvbench PUBLIC nvbench::nvbench)
target_link_libraries(_nvbench PRIVATE CUDA::cudart_static)

set_target_properties(_nvbench PROPERTIES INSTALL_RPATH "$ORIGIN")
set_target_properties(_nvbench PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)
set_target_properties(_nvbench PROPERTIES POSITION_INDEPENDENT_CODE ON)

install(TARGETS _nvbench DESTINATION cuda/bench)

# Determine target that nvbench::nvbench is an alias of,
# necessary because ALIAS targets cannot be installed
get_target_property(_aliased_target_name nvbench::nvbench ALIASED_TARGET)
install(IMPORTED_RUNTIME_ARTIFACTS ${_aliased_target_name} DESTINATION cuda/bench)
