set(srcs
  axes_metadata.cxx
  axis_base.cxx
  benchmark_base.cxx
  benchmark_manager.cxx
  blocking_kernel.cu
  csv_printer.cu
  cuda_call.cu
  device_info.cu
  device_manager.cu
  float64_axis.cxx
  int64_axis.cxx
  markdown_printer.cu
  named_values.cxx
  option_parser.cu
  printer_base.cxx
  printer_multiplex.cxx
  runner.cxx
  state.cxx
  string_axis.cxx
  type_axis.cxx
  type_strings.cxx

  detail/measure_cold.cu
  detail/measure_hot.cu
  detail/state_generator.cxx
)

if (NVBench_ENABLE_CUPTI)
  list(APPEND srcs detail/measure_cupti.cu cupti_profiler.cxx)
endif()

if (NVBench_ENABLE_NVML)
  list(APPEND srcs internal/nvml.cxx)
endif()

# CUDA 11.0 can't compile json_printer without crashing
# So for that version fall back to C++ with degraded
# output ( no PTX version info )
if(CMAKE_CUDA_COMPILER_ID STREQUAL NVIDIA AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.1)
  set(json_printer_impl json_printer.cxx)
  set(json_is_cu FALSE)
else()
  set(json_printer_impl json_printer.cu)
  set(json_is_cu TRUE)
endif()
list(APPEND srcs ${json_printer_impl})

# Generate doc strings from md files:
include("../cmake/FileToString.cmake")
file_to_string("../docs/cli_help.md"
  "${NVBench_BINARY_DIR}/nvbench/internal/cli_help.cuh"
  ""
  cli_help_text
)
file_to_string("../docs/cli_help_axis.md"
  "${NVBench_BINARY_DIR}/nvbench/internal/cli_help_axis.cuh"
  ""
  cli_help_axis_text
)

nvbench_write_config_header(config.cuh.in
  "${NVBench_BINARY_DIR}/nvbench/config.cuh"
)

# nvbench (nvbench::nvbench)
add_library(nvbench SHARED ${srcs})
nvbench_config_target(nvbench)
target_include_directories(nvbench PUBLIC
  "$<BUILD_INTERFACE:${NVBench_SOURCE_DIR}>"
  "$<BUILD_INTERFACE:${NVBench_BINARY_DIR}>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)
target_link_libraries(nvbench
  PUBLIC
    ${ctk_libraries}
  PRIVATE
    nvbench_json
    nvbench_git_revision
)

# ##################################################################################################
# * conda environment -----------------------------------------------------------------------------
rapids_cmake_support_conda_env(conda_env MODIFY_PREFIX_PATH)
if(TARGET conda_env)
  # When we are inside a conda env the linker will be set to
  # `ld.bfd` which will try to resolve all undefined symbols at link time.
  #
  # Since we could be using a shared library version of fmt we need
  # it on the final link line of consumers
  target_link_libraries(nvbench PRIVATE $<BUILD_INTERFACE:conda_env>)
endif()


# When we are inside a conda env the linker will be set to
# `ld.bfd` which will try to resolve all undefined symbols at link time.
#
# Since we could be using a shared library version of fmt we need
# it on the final link line of consumers
if(TARGET conda_env AND fmt_is_external)
  target_link_libraries(nvbench PUBLIC fmt::fmt)
else()
  target_link_libraries(nvbench PRIVATE fmt::fmt)
endif()

target_compile_features(nvbench PUBLIC cuda_std_17 PRIVATE cxx_std_17)
add_dependencies(nvbench.all nvbench)

# nvbench.main (nvbench::main)
add_library(nvbench.main OBJECT main.cu)
nvbench_config_target(nvbench.main)
target_link_libraries(nvbench.main PUBLIC nvbench)
set_target_properties(nvbench.main PROPERTIES EXPORT_NAME main)
add_dependencies(nvbench.all nvbench.main)

# Support add_subdirectory:
add_library(nvbench::nvbench ALIAS nvbench)
add_library(nvbench::main ALIAS nvbench.main)

nvbench_setup_dep_dlls(nvbench)
nvbench_install_libraries(nvbench nvbench.main nvbench.build_interface)

# nvcc emits several unavoidable warnings while compiling nlohmann_json:
if (json_is_cu)
  set_property(SOURCE ${json_printer_impl} APPEND PROPERTY COMPILE_OPTIONS
    # error #186-D: pointless comparison of unsigned integer with zero
    $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcudafe=--diag_suppress=186>
    # error #940-D: missing return statement at end of non-void function
    # (the end of the function in hash.hpp(114) is unreachable)
    $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcudafe=--diag_suppress=940>
  )
endif()
