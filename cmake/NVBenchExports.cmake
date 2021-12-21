macro(nvbench_generate_exports)
  set(nvbench_build_export_code_block "")
  set(nvbench_install_export_code_block "")

  if (NVBench_ENABLE_NVML)
    string(APPEND nvbench_build_export_code_block
      "include(\"${NVBench_SOURCE_DIR}/cmake/NVBenchNVML.cmake\")\n"
    )
    string(APPEND nvbench_install_export_code_block
      "include(\"\${CMAKE_CURRENT_LIST_DIR}/NVBenchNVML.cmake\")\n"
    )
  endif()

  if (NVBench_ENABLE_CUPTI)
    string(APPEND nvbench_build_export_code_block
      "include(\"${NVBench_SOURCE_DIR}/cmake/NVBenchCUPTI.cmake\")\n"
    )
    string(APPEND nvbench_install_export_code_block
      "include(\"\${CMAKE_CURRENT_LIST_DIR}/NVBenchCUPTI.cmake\")\n"
    )
  endif()

  rapids_export(BUILD NVBench
    EXPORT_SET nvbench-targets
    NAMESPACE "nvbench::"
    GLOBAL_TARGETS nvbench main ctl internal_build_interface
    LANGUAGES CUDA CXX
    FINAL_CODE_BLOCK nvbench_build_export_code_block
  )
  rapids_export(INSTALL NVBench
    EXPORT_SET nvbench-targets
    NAMESPACE "nvbench::"
    GLOBAL_TARGETS nvbench main ctl internal_build_interface
    LANGUAGES CUDA CXX
    FINAL_CODE_BLOCK nvbench_install_export_code_block
  )
endmacro()
