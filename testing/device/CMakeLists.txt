# Test that we're converging to an accurate mean + stdev without timing out:
set(test_name nvbench.test.device.noisy_bench)
add_executable(${test_name} noisy_bench.cu)
target_link_libraries(${test_name} PRIVATE nvbench::main fmt::fmt)
nvbench_config_target(${test_name})
add_dependencies(nvbench.test.all ${test_name})

if (NVBench_ENABLE_DEVICE_TESTING)
  add_test(NAME ${test_name} COMMAND "$<TARGET_FILE:${test_name}>")
  set_tests_properties(${test_name} PROPERTIES
    # Any timeouts/warnings are hard failures for this test.
    FAIL_REGULAR_EXPRESSION "Warn;timed out"
  )
endif()

set(test_name nvbench.test.device.exception_safety)
add_executable(${test_name} exception_safety.cu)
target_include_directories(${test_name} PRIVATE "${CMAKE_CURRENT_LIST_DIR}/..")
target_link_libraries(${test_name} PRIVATE nvbench::nvbench fmt::fmt)
set_target_properties(${test_name} PROPERTIES COMPILE_FEATURES cuda_std_17)
nvbench_config_target(${test_name})
add_dependencies(nvbench.test.all ${test_name})

if (NVBench_ENABLE_DEVICE_TESTING)
  add_test(NAME ${test_name} COMMAND "$<TARGET_FILE:${test_name}>")
  set_tests_properties(${test_name} PROPERTIES TIMEOUT 20)
endif()
