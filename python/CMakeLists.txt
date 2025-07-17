cmake_minimum_required(VERSION 3.30...4.0)

project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python REQUIRED COMPONENTS Development.Module)
find_package(CUDAToolkit REQUIRED)

include(FetchContent)

FetchContent_Declare(
    pybind11
    URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.tar.gz
    URL_HASH SHA256=e08cb87f4773da97fa7b5f035de8763abc656d87d5773e62f6da0587d1f0ec20
    FIND_PACKAGE_ARGS NAMES pybind11
)
FetchContent_MakeAvailable(pybind11)

find_package(nvbench CONFIG REQUIRED)

pybind11_add_module(_nvbench MODULE src/py_nvbench.cpp)
target_link_libraries(_nvbench PUBLIC nvbench::nvbench)
target_link_libraries(_nvbench PRIVATE CUDA::cudart_static)
set_target_properties(_nvbench PROPERTIES INSTALL_RPATH "$ORIGIN")

install(TARGETS _nvbench DESTINATION cuda/nvbench)
install(IMPORTED_RUNTIME_ARTIFACTS nvbench::nvbench DESTINATION cuda/nvbench)
