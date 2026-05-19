add_executable(nvbench.ctl nvbench-ctl.cu)
nvbench_config_target(nvbench.ctl)
target_link_libraries(nvbench.ctl PRIVATE nvbench)
set_target_properties(nvbench.ctl PROPERTIES
  OUTPUT_NAME nvbench-ctl
  EXPORT_NAME ctl
)
add_dependencies(nvbench.all nvbench.ctl)
nvbench_install_executables(nvbench.ctl)

if (NVBench_ENABLE_TESTING)
  set(ctl_test_names)

  # Test: nvbench
  set(test_name nvbench.ctl.no_args)
  add_test(NAME ${test_name} COMMAND "$<TARGET_FILE:nvbench.ctl>")
  list(APPEND ctl_test_names ${test_name})
  # Should print the version without any args:
  set_property(TEST ${test_name}
    PROPERTY PASS_REGULAR_EXPRESSION "NVBench v"
  )

  # Test: nvbench --version
  set(test_name nvbench.ctl.version)
  add_test(NAME ${test_name} COMMAND "$<TARGET_FILE:nvbench.ctl>" --version)
  list(APPEND ctl_test_names ${test_name})
  # Should print the version without any args:
  set_property(TEST ${test_name}
    PROPERTY PASS_REGULAR_EXPRESSION "NVBench v"
  )

  # Test: nvbench --list
  set(test_name nvbench.ctl.list)
  add_test(NAME ${test_name} COMMAND "$<TARGET_FILE:nvbench.ctl>" --list)
  list(APPEND ctl_test_names ${test_name})

  # Test: nvbench -l
  set(test_name nvbench.ctl.l)
  add_test(NAME ${test_name} COMMAND "$<TARGET_FILE:nvbench.ctl>" -l)
  list(APPEND ctl_test_names ${test_name})

  # Test: nvbench --help
  set(test_name nvbench.ctl.help)
  add_test(NAME ${test_name} COMMAND "$<TARGET_FILE:nvbench.ctl>" --help)
  list(APPEND ctl_test_names ${test_name})

  # Test: nvbench -h
  set(test_name nvbench.ctl.h)
  add_test(NAME ${test_name} COMMAND "$<TARGET_FILE:nvbench.ctl>" -h)
  list(APPEND ctl_test_names ${test_name})

  # Test: nvbench --help-axes
  set(test_name nvbench.ctl.help_axes)
  add_test(NAME ${test_name} COMMAND "$<TARGET_FILE:nvbench.ctl>" --help-axes)
  list(APPEND ctl_test_names ${test_name})

  # Test: nvbench --help-axis
  set(test_name nvbench.ctl.help_axis)
  add_test(NAME ${test_name} COMMAND "$<TARGET_FILE:nvbench.ctl>" --help-axis)
  list(APPEND ctl_test_names ${test_name})

  foreach(test_name IN LISTS ctl_test_names)
    nvbench_config_test_runtime_environment(${test_name})
  endforeach()
endif()
