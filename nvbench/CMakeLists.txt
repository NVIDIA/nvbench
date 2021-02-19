set(srcs
  axes_metadata.cu
  axis_base.cu
  benchmark_base.cu
  benchmark_manager.cu
  blocking_kernel.cu
  cuda_call.cu
  device_info.cu
  device_manager.cu
  float64_axis.cu
  int64_axis.cu
  named_values.cu
  option_parser.cu
  runner.cu
  state.cu
  string_axis.cu
  type_axis.cu
  type_strings.cu

  detail/markdown_format.cu
  detail/measure_cold.cu
  detail/measure_hot.cu
  detail/state_generator.cu
)

add_library(nvbench STATIC ${srcs})
add_library(nvbench::nvbench ALIAS nvbench)

# TODO generator expressions for installed paths
target_include_directories(nvbench PUBLIC "${nvbench_SOURCE_DIR}")
target_link_libraries(nvbench PRIVATE fmt::fmt)
target_compile_features(nvbench PUBLIC cuda_std_17)

add_library(nvbench_main OBJECT main.cu)
add_library(nvbench::main ALIAS nvbench_main)
target_link_libraries(nvbench_main PUBLIC nvbench)
