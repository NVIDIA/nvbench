# By default, add dependent DLLs to the build dir on MSVC. This avoids
# a variety of runtime issues when using NVML, etc.
# This behavior can be disabled using the following options:
if (WIN32)
  option(NVBench_ADD_DEPENDENT_DLLS_TO_BUILD
    "Copy dependent dlls to NVBench library build location (MSVC only)."
    ON
  )
else()
  # These are forced off for non-MSVC builds, as $<TARGET_RUNTIME_DLLS:...>
  # will always be empty on non-dll platforms.
  set(NVBench_ADD_DEPENDENT_DLLS_TO_BUILD OFF)
endif()

function(nvbench_setup_dep_dlls target_name)
  # The custom command below fails when there aren't any runtime DLLs to copy,
  # so only enable it when a relevant dependency is enabled:
  if (NVBench_ADD_DEPENDENT_DLLS_TO_BUILD AND
      (NVBench_ENABLE_NVML OR
       NVBench_ENABLE_CUPTI))
    add_custom_command(TARGET ${target_name}
      POST_BUILD
      COMMAND
        "${CMAKE_COMMAND}" -E copy
          "$<TARGET_RUNTIME_DLLS:${target_name}>"
          "$<TARGET_FILE_DIR:${target_name}>"
      COMMAND_EXPAND_LISTS
    )
  endif()
endfunction()
