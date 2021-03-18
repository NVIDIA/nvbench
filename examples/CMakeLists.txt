set(example_srcs
  axes.cu
  enums.cu
  exec_tag_sync.cu
  exec_tag_timer.cu
  skip.cu
  throughput.cu
)

# Metatarget for all examples:
add_custom_target(nvbench.example.all)
add_dependencies(nvbench.all nvbench.example.all)

foreach(example_src IN LISTS example_srcs)
  get_filename_component(example_name "${example_src}" NAME_WLE)
  string(PREPEND example_name "nvbench.example.")
  add_executable(${example_name} "${example_src}")
  target_include_directories(${example_name} PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
  target_link_libraries(${example_name} PRIVATE nvbench::main)
  set_target_properties(${example_name} PROPERTIES COMPILE_FEATURES cuda_std_17)
  set_target_properties(${example_name} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${NVBench_LIBRARY_OUTPUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${NVBench_LIBRARY_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${NVBench_EXECUTABLE_OUTPUT_DIR}"
  )
  add_test(NAME ${example_name}
    COMMAND "$<TARGET_FILE:${example_name}>" --timeout 1
  )

  add_dependencies(nvbench.example.all ${example_name})
endforeach()
