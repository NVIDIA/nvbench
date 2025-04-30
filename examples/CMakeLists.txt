set(example_srcs
  auto_throughput.cu
  axes.cu
  custom_criterion.cu
  cpu_only.cu
  enums.cu
  exec_tag_sync.cu
  exec_tag_timer.cu
  skip.cu
  stream.cu
  summaries.cu
  throughput.cu
)

# Metatarget for all examples:
add_custom_target(nvbench.example.all)
add_dependencies(nvbench.all nvbench.example.all)

function (nvbench_add_examples_target target_prefix cuda_std)
  add_custom_target(${target_prefix}.all)
  add_dependencies(nvbench.example.all ${target_prefix}.all)

  foreach(example_src IN LISTS example_srcs)
    get_filename_component(example_name "${example_src}" NAME_WLE)
    string(PREPEND example_name "${target_prefix}.")
    add_executable(${example_name} "${example_src}")
    nvbench_config_target(${example_name})
    target_include_directories(${example_name} PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
    target_link_libraries(${example_name} PRIVATE nvbench::main)
    set_target_properties(${example_name} PROPERTIES COMPILE_FEATURES cuda_std_${cuda_std})

    set(example_args --timeout 0.1)
    # The custom_criterion example doesn't support the --min-time argument:
    if (NOT "${example_src}" STREQUAL "custom_criterion.cu")
      list(APPEND example_args --min-time 1e-5)
    endif()

    add_test(NAME ${example_name}
      COMMAND "$<TARGET_FILE:${example_name}>" ${example_args})

    # These should not deadlock. If they do, it may be that the CUDA context was created before
    # setting CUDA_MODULE_LOAD=EAGER in main, see NVIDIA/nvbench#136.
    set_tests_properties(${example_name} PROPERTIES
      FAIL_REGULAR_EXPRESSION "Possible Deadlock Detected"
    )

    add_dependencies(${target_prefix}.all ${example_name})
  endforeach()
endfunction()


foreach (std IN LISTS NVBench_DETECTED_CUDA_STANDARDS)
  nvbench_add_examples_target(nvbench.example.cpp${std} ${std})
endforeach()
