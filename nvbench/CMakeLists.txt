set(srcs
  axis_base.cu
  type_axis.cu
  int64_axis.cu
)

# TODO shared may be a good idea to reduce compilation overhead for large
# benchmark libraries.
add_library(nvbench STATIC ${srcs})
# TODO generator expression for installed paths
target_include_directories(nvbench PUBLIC "${nvbench_SOURCE_DIR}")
target_link_libraries(nvbench PRIVATE fmt::fmt)
target_link_libraries(nvbench INTERFACE benchmark_main)
set_target_properties(nvbench PROPERTIES COMPILE_FEATURES cuda_std_17)
