cmake_minimum_required(VERSION 3.30.4)
project(nvbench_test_dep_dlls CXX)
enable_testing()

# Simulate the Windows helper branch on every host without CUDA or a driver.
# The fixture has no shared dependencies, so TARGET_RUNTIME_DLLS is empty.
set(WIN32 TRUE)
set(MSVC TRUE)
set(NVBench_ENABLE_NVML ON)
set(NVBench_ENABLE_CUPTI OFF)
set(NVBench_ADD_DEPENDENT_DLLS_TO_BUILD ON)
set(NVBench_NVML_DLL "")
set(expect_copy OFF)

if (NOT COPY_CASE STREQUAL "empty")
  if (COPY_CASE STREQUAL "spaces")
    set(driver_dir "${CMAKE_CURRENT_BINARY_DIR}/Program Files/NVIDIA Corporation")
  else()
    set(driver_dir "${CMAKE_CURRENT_BINARY_DIR}/driver&special")
  endif()
  file(MAKE_DIRECTORY "${driver_dir}")
  set(NVBench_NVML_DLL "${driver_dir}/nvml.dll")
  file(WRITE "${NVBench_NVML_DLL}" "NVML copy test fixture\n")
  set(expect_copy ON)
endif()

if (COPY_CASE STREQUAL "disabled")
  set(NVBench_ADD_DEPENDENT_DLLS_TO_BUILD OFF)
  set(expect_copy OFF)
endif()

if (COPY_CASE STREQUAL "static")
  add_library(copy_fixture STATIC main.cxx)
  set(expect_copy OFF)
else()
  add_executable(copy_fixture main.cxx)
endif()

include("${NVBench_SOURCE_DIR}/cmake/NVBenchDependentDlls.cmake")
nvbench_setup_dep_dlls(copy_fixture)

add_test(NAME check_copy
  COMMAND "${CMAKE_COMMAND}"
    "-DEXPECT_COPY=${expect_copy}"
    "-DINPUT_FILE=${NVBench_NVML_DLL}"
    "-DOUTPUT_FILE=$<TARGET_FILE_DIR:copy_fixture>/nvml.dll"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/check_copy.cmake"
)
