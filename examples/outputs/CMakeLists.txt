set(NVBench_EXAMPLE_OUTPUT_DEVICES "all" CACHE STRING
  "--devices argument used when generating example outputs."
)
set(NVBench_EXAMPLE_OUTPUT_TIMEOUT "15.0" CACHE STRING
  "--timeout argument used when generating example outputs."
)
mark_as_advanced(
  NVBench_EXAMPLE_OUTPUT_DEVICES
  NVBench_EXAMPLE_OUTPUT_TIMEOUT
)

set_property(GLOBAL PROPERTY JOB_POOLS exclusive=1)

add_custom_target(nvbench.regenerate_example_outputs)
add_custom_target(nvbench.example.all.list)

set(results_dir "${CMAKE_CURRENT_BINARY_DIR}/results")

foreach (example IN LISTS examples)
  # Print --list
  add_custom_target(${example}.list
    COMMAND "$<TARGET_FILE:${example}>"
      --md "${results_dir}/${example}.list.md"
      --list
    BYPRODUCTS "${results_dir}/${example}.list.md"
    COMMENT "Generating ${example}.list.md..."
    VERBATIM
  )
  add_dependencies(${example}.list nvbench.example.all)
  add_dependencies(nvbench.regenerate_example_outputs ${example}.list)
  add_dependencies(nvbench.example.all.list ${example}.list)

  # Run and output all formats
  add_custom_target(${example}.outputs
    COMMAND "$<TARGET_FILE:${example}>"
      --quiet
      --devices ${NVBench_EXAMPLE_OUTPUT_DEVICES}
      --timeout ${NVBench_EXAMPLE_OUTPUT_TIMEOUT}
      --md "${results_dir}/${example}.md"
      --csv "${results_dir}/${example}.csv"
      --json "${results_dir}/${example}.json"
    BYPRODUCTS
      "${results_dir}/${example}.md"
      "${results_dir}/${example}.csv"
      "${results_dir}/${example}.json"
    COMMENT "Generating ${example} outputs (json, csv, md)..."
    JOB_POOL exclusive
    VERBATIM
  )
  add_dependencies(${example}.outputs nvbench.example.all)
  add_dependencies(nvbench.regenerate_example_outputs ${example}.outputs)
endforeach()
