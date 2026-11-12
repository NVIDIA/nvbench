# By default, add dependent DLLs to the build directory on Windows. This
# avoids runtime lookup failures for NVML, CUPTI, and their dependencies.
if (WIN32 AND MSVC)
  option(NVBench_ADD_DEPENDENT_DLLS_TO_BUILD
    "Copy dependent DLLs to the NVBench build directories."
    ON
  )
else()
  # TARGET_RUNTIME_DLLS is only useful for Windows DLL targets. Keep the
  # option disabled elsewhere so this helper has no effect on other platforms.
  set(NVBench_ADD_DEPENDENT_DLLS_TO_BUILD OFF)
endif()

function(nvbench_setup_dep_dlls target_name)
  get_target_property(target_type "${target_name}" TYPE)

  # TARGET_RUNTIME_DLLS is only valid for executables and shared/module
  # libraries. Static libraries are covered by their final consumer instead.
  if (NVBench_ADD_DEPENDENT_DLLS_TO_BUILD AND
      (NVBench_ENABLE_NVML OR NVBench_ENABLE_CUPTI) AND
      (target_type STREQUAL "EXECUTABLE" OR
        target_type STREQUAL "SHARED_LIBRARY" OR
        target_type STREQUAL "MODULE_LIBRARY"))
    set(runtime_dlls "$<TARGET_RUNTIME_DLLS:${target_name}>")
    if (NVBench_ENABLE_NVML AND NVBench_NVML_DLL)
      # CUDA::nvml is an UNKNOWN imported target and is omitted by
      # TARGET_RUNTIME_DLLS, so add its driver-installed DLL explicitly when
      # it is available on the build machine.
      list(APPEND runtime_dlls "${NVBench_NVML_DLL}")
    endif()

    # Targets in different directories can share a runtime output directory.
    # A per-configuration lock prevents parallel post-build steps from copying
    # the same DLL to that directory at the same time.
    set(runtime_dlls_manifest "${CMAKE_CURRENT_BINARY_DIR}/${target_name}_runtime_dlls-$<CONFIG>.txt")
    file(GENERATE
      OUTPUT "${runtime_dlls_manifest}"
      CONTENT "${runtime_dlls}"
    )

    add_custom_command(TARGET ${target_name}
      POST_BUILD
      COMMAND "${CMAKE_COMMAND}"
        "-DNVBENCH_RUNTIME_DLLS_FILE=${runtime_dlls_manifest}"
        "-DNVBENCH_RUNTIME_DLLS_DEST=$<TARGET_FILE_DIR:${target_name}>"
        "-DNVBENCH_RUNTIME_DLLS_LOCK=${CMAKE_BINARY_DIR}/CMakeFiles/nvbench-runtime-dll-copy-$<CONFIG>.lock"
        -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/NVBenchCopyRuntimeDlls.cmake"
      VERBATIM
    )
  endif()
endfunction()
