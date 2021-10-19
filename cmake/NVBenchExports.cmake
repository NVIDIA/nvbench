macro(nvbench_generate_exports)
  rapids_export(BUILD NVBench
    EXPORT_SET nvbench-targets
    NAMESPACE "nvbench::"
    GLOBAL_TARGETS nvbench main
    LANGUAGES CUDA CXX
  )
  rapids_export(INSTALL NVBench
    EXPORT_SET nvbench-targets
    NAMESPACE "nvbench::"
    GLOBAL_TARGETS nvbench main
    LANGUAGES CUDA CXX
  )
endmacro()
