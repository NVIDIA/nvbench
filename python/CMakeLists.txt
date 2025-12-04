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

# Determine CUDA major version for directory structure
string(REGEX MATCH "^([0-9]+)" CUDA_VERSION_MAJOR "${CUDAToolkit_VERSION}")
set(CUDA_VERSION_DIR "cu${CUDA_VERSION_MAJOR}")
message(STATUS "Building extension for CUDA ${CUDA_VERSION_MAJOR}, output directory: cuda/bench/${CUDA_VERSION_DIR}")

# Extension name is always _nvbench (directory provides CUDA version separation)
set(NVBENCH_EXTENSION_NAME "_nvbench")

pybind11_add_module(${NVBENCH_EXTENSION_NAME} MODULE src/py_nvbench.cpp)
target_compile_definitions(${NVBENCH_EXTENSION_NAME} PRIVATE PYBIND11_MODULE_NAME=${NVBENCH_EXTENSION_NAME})
target_link_libraries(${NVBENCH_EXTENSION_NAME} PUBLIC nvbench::nvbench)
target_link_libraries(${NVBENCH_EXTENSION_NAME} PRIVATE CUDA::cudart_static CUDA::cuda_driver)

set_target_properties(${NVBENCH_EXTENSION_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN")
set_target_properties(${NVBENCH_EXTENSION_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)
set_target_properties(${NVBENCH_EXTENSION_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

install(TARGETS ${NVBENCH_EXTENSION_NAME} DESTINATION cuda/bench/${CUDA_VERSION_DIR})

# Determine target that nvbench::nvbench is an alias of,
# necessary because ALIAS targets cannot be installed
get_target_property(_aliased_target_name nvbench::nvbench ALIASED_TARGET)
install(IMPORTED_RUNTIME_ARTIFACTS ${_aliased_target_name} DESTINATION cuda/bench/${CUDA_VERSION_DIR})
