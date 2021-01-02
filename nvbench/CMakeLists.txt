set(srcs
  axes_metadata.cu
  axis_base.cu
  benchmark_base.cu
  benchmark_manager.cu
  cuda_call.cu
  float64_axis.cu
  int64_axis.cu
  named_values.cu
  state.cu
  string_axis.cu
  type_axis.cu

  detail/markdown_format.cu
  detail/measure_hot.cu
  detail/state_generator.cu
)

# TODO shared may be a good idea to reduce compilation overhead for large
# benchmark libraries.
add_library(nvbench STATIC ${srcs})
# TODO generator expression for installed paths
target_include_directories(nvbench PUBLIC "${nvbench_SOURCE_DIR}")
target_link_libraries(nvbench PRIVATE fmt::fmt)
target_link_libraries(nvbench INTERFACE benchmark_main)
set_target_properties(nvbench PROPERTIES COMPILE_FEATURES cuda_std_17)
