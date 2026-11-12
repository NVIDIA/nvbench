cmake_minimum_required(VERSION 3.30.4)
project(nvbench_test_dep_dlls CXX)
enable_testing()

set(native_windows ${WIN32})
if (COPY_CASE MATCHES "^static_consumer")
  # Exercise the same target setup used by examples and tests. Include it
  # before simulating Windows so compiler checks still use the actual host.
  include("${NVBench_SOURCE_DIR}/cmake/NVBenchConfigTarget.cmake")
endif()

# Simulate the Windows helper branch on every host without CUDA or a driver.
# TARGET_RUNTIME_DLLS is empty on non-Windows hosts.
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

if (COPY_CASE STREQUAL "disabled" OR COPY_CASE STREQUAL "static_consumer_disabled")
  set(NVBench_ADD_DEPENDENT_DLLS_TO_BUILD OFF)
  set(expect_copy OFF)
endif()

if (COPY_CASE STREQUAL "static_consumer_cupti")
  set(NVBench_ENABLE_NVML OFF)
  set(NVBench_ENABLE_CUPTI ON)
  set(expect_copy OFF)
endif()

if (COPY_CASE STREQUAL "static")
  add_library(copy_fixture STATIC main.cxx)
  set(expect_copy OFF)
else()
  add_executable(copy_fixture main.cxx)
endif()

include("${NVBench_SOURCE_DIR}/cmake/NVBenchDependentDlls.cmake")
if (COPY_CASE STREQUAL "parallel")
  foreach(target_number RANGE 1 8)
    set(target_name "copy_fixture_parallel_${target_number}")
    add_executable(${target_name} main.cxx)
    nvbench_setup_dep_dlls(${target_name})
  endforeach()
endif()

if (COPY_CASE MATCHES "^static_consumer")
  # Keep the dependency DLL outside the executable directory. No control
  # executable or shared NVBench library can supply the missing copy step.
  add_library(runtime_dependency SHARED runtime_dependency.cxx)
  set_target_properties(runtime_dependency PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/dependencies"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/dependencies"
  )
  add_library(static_dependency STATIC static_dependency.cxx)
  target_link_libraries(static_dependency PRIVATE runtime_dependency)
  nvbench_config_target(static_dependency)
  target_link_libraries(copy_fixture PRIVATE static_dependency)
  target_compile_definitions(copy_fixture PRIVATE STATIC_CONSUMER)
  nvbench_config_target(copy_fixture)

  if (native_windows)
    add_test(NAME check_transitive_dll
      COMMAND "${CMAKE_COMMAND}"
        "-DEXPECT_COPY=${NVBench_ADD_DEPENDENT_DLLS_TO_BUILD}"
        "-DINPUT_FILE=$<TARGET_FILE:runtime_dependency>"
        "-DOUTPUT_FILE=$<TARGET_FILE_DIR:copy_fixture>/$<TARGET_FILE_NAME:runtime_dependency>"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/check_copy.cmake"
    )
  endif()
  if (NVBench_ADD_DEPENDENT_DLLS_TO_BUILD)
    add_test(NAME run_static_consumer COMMAND copy_fixture)
  endif()
else()
  nvbench_setup_dep_dlls(copy_fixture)
endif()

add_test(NAME check_copy
  COMMAND "${CMAKE_COMMAND}"
    "-DEXPECT_COPY=${expect_copy}"
    "-DINPUT_FILE=${NVBench_NVML_DLL}"
    "-DOUTPUT_FILE=$<TARGET_FILE_DIR:copy_fixture>/nvml.dll"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/check_copy.cmake"
)
