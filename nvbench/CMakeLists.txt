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

# CUDA 11.0 can't compile json_printer without crashing
# So for that version fall back to C++ with degraded
# output ( no PTX version info )
if(CMAKE_CUDA_COMPILER_ID STREQUAL NVIDIA AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.1)
  list(APPEND srcs json_printer.cxx)
else()
  list(APPEND srcs json_printer.cu)
endif()

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

# nvbench (nvbench::nvbench)
add_library(nvbench SHARED ${srcs})
target_include_directories(nvbench PUBLIC
  "$<BUILD_INTERFACE:${NVBench_SOURCE_DIR}>"
  "$<BUILD_INTERFACE:${NVBench_BINARY_DIR}>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)
target_link_libraries(nvbench PUBLIC ${ctk_libraries})
target_link_libraries(nvbench PRIVATE
  fmt::fmt
  nvbench_json
  nvbench_git_revision
)
nvbench_config_target(nvbench)
target_compile_features(nvbench PUBLIC cuda_std_17 PRIVATE cxx_std_17)
add_dependencies(nvbench.all nvbench)

# nvbench_main (nvbench::main)
add_library(nvbench_main OBJECT main.cu)
nvbench_config_target(nvbench_main)
target_link_libraries(nvbench_main PUBLIC nvbench)
set_target_properties(nvbench_main PROPERTIES EXPORT_NAME main)
add_dependencies(nvbench.all nvbench_main)

# Support add_subdirectory:
add_library(nvbench::nvbench ALIAS nvbench)
add_library(nvbench::main ALIAS nvbench_main)

nvbench_install_libraries(nvbench nvbench_main)
