macro(nvbench_generate_exports)
  if(NVBench_ENABLE_INSTALL_RULES)
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

    if (TARGET nvbench_json)
      set(nvbench_json_code_block
        [=[
        add_library(nvbench_json INTERFACE IMPORTED)
        if (TARGET nlohmann_json::nlohmann_json)
          target_link_libraries(nvbench_json INTERFACE nlohmann_json::nlohmann_json)
        endif()
        ]=])
      string(APPEND nvbench_build_export_code_block ${nvbench_json_code_block})
      string(APPEND nvbench_install_export_code_block ${nvbench_json_code_block})
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
  endif()
endmacro()
