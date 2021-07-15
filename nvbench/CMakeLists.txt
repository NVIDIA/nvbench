set(srcs
  axes_metadata.cu
  axis_base.cu
  benchmark_base.cu
  benchmark_manager.cu
  blocking_kernel.cu
  csv_printer.cu
  cuda_call.cu
  device_info.cu
  device_manager.cu
  float64_axis.cu
  int64_axis.cu
  markdown_printer.cu
  named_values.cu
  option_parser.cu
  printer_base.cu
  printer_multiplex.cu
  runner.cu
  state.cu
  string_axis.cu
  type_axis.cu
  type_strings.cu

  detail/measure_cold.cu
  detail/measure_hot.cu
  detail/state_generator.cu
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
  "${CMAKE_CURRENT_BINARY_DIR}/cli_help.cuh"
  ""
  cli_help_text
)
file_to_string("../docs/cli_help_axis.md"
  "${CMAKE_CURRENT_BINARY_DIR}/cli_help_axis.cuh"
  ""
  cli_help_axis_text
)

find_package(CUDAToolkit)

add_library(nvbench STATIC ${srcs})
add_library(nvbench::nvbench ALIAS nvbench)
# TODO generator expressions for installed paths
target_include_directories(nvbench PUBLIC "${NVBench_SOURCE_DIR}")
target_include_directories(nvbench PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_include_directories(nvbench PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")
target_link_libraries(nvbench PRIVATE fmt::fmt nvbench_json)
target_compile_features(nvbench PUBLIC cuda_std_17 PRIVATE cxx_std_17)
set_target_properties(nvbench PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${NVBench_LIBRARY_OUTPUT_DIR}"
  LIBRARY_OUTPUT_DIRECTORY "${NVBench_LIBRARY_OUTPUT_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${NVBench_EXECUTABLE_OUTPUT_DIR}"
)
add_dependencies(nvbench.all nvbench)

add_library(nvbench_main OBJECT main.cu)
add_library(nvbench::main ALIAS nvbench_main)
target_link_libraries(nvbench_main PUBLIC nvbench)
set_target_properties(nvbench PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${NVBench_LIBRARY_OUTPUT_DIR}"
  LIBRARY_OUTPUT_DIRECTORY "${NVBench_LIBRARY_OUTPUT_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${NVBench_EXECUTABLE_OUTPUT_DIR}"
)
add_dependencies(nvbench.all nvbench_main)
