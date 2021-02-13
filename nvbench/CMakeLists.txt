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
  state.cu
  string_axis.cu
  type_axis.cu

  detail/markdown_format.cu
  detail/measure_cold.cu
  detail/measure_hot.cu
  detail/state_generator.cu
)

add_library(nvbench STATIC ${srcs})

# TODO generator expressions for installed paths
target_include_directories(nvbench PUBLIC "${nvbench_SOURCE_DIR}")
target_link_libraries(nvbench PRIVATE fmt::fmt)
target_link_libraries(nvbench INTERFACE benchmark_main)
target_compile_features(nvbench PUBLIC cuda_std_17)
