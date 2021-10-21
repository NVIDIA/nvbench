function(nvbench_write_config_header filepath)
  if (NVBench_ENABLE_NVML)
    set(NVBENCH_HAS_NVML 1)
  endif()

  configure_file("${NVBench_SOURCE_DIR}/cmake/config.cuh.in" "${filepath}")
endfunction()
