set(test_srcs
  type_list.cu
)

foreach(test_src IN LISTS test_srcs)
  get_filename_component(test_name "${test_src}" NAME_WLE)
  string(PREPEND test_name "nvbench.test.")
  add_executable(${test_name} "${test_src}")
  target_link_libraries(${test_name} PRIVATE nvbench)
  set_target_properties(${test_name} PROPERTIES COMPILE_FEATURES cuda_std_17)
  add_test(NAME ${test_name} COMMAND "$<TARGET_FILE:${test_target}>")
endforeach()
