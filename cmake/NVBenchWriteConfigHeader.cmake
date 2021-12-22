function(nvbench_write_config_header in_file out_file)
  if (NVBench_ENABLE_NVML)
    set(NVBENCH_HAS_NVML 1)
  endif()

  if (NVBench_ENABLE_CUPTI)
    set(NVBENCH_HAS_CUPTI 1)
  endif()

  configure_file("${in_file}" "${out_file}")
endfunction()
