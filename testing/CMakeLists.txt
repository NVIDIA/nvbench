set(test_srcs
  axes_metadata.cu
  benchmark.cu
  create.cu
  cuda_timer.cu
  cpu_timer.cu
  enum_type_list.cu
  float64_axis.cu
  int64_axis.cu
  named_values.cu
  option_parser.cu
  range.cu
  runner.cu
  state.cu
  state_generator.cu
  string_axis.cu
  type_axis.cu
  type_list.cu
)

# Metatarget for all examples:
add_custom_target(nvbench.test.all)

foreach(test_src IN LISTS test_srcs)
  get_filename_component(test_name "${test_src}" NAME_WLE)
  string(PREPEND test_name "nvbench.test.")
  add_executable(${test_name} "${test_src}")
  target_include_directories(${test_name} PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
  target_link_libraries(${test_name} PRIVATE nvbench::nvbench fmt)
  set_target_properties(${test_name} PROPERTIES COMPILE_FEATURES cuda_std_17)
  set_target_properties(${test_name} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${NVBench_LIBRARY_OUTPUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${NVBench_LIBRARY_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${NVBench_EXECUTABLE_OUTPUT_DIR}"
  )
  add_test(NAME ${test_name} COMMAND "$<TARGET_FILE:${test_name}>")

  add_dependencies(nvbench.test.all ${test_name})
endforeach()
