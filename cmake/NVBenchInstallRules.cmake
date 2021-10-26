include(GNUInstallDirs)
rapids_cmake_install_lib_dir(NVBench_INSTALL_LIB_DIR)

# in-source public headers:
install(DIRECTORY "${NVBench_SOURCE_DIR}/nvbench"
  TYPE INCLUDE
  FILES_MATCHING
    PATTERN "*.cuh"
    PATTERN "internal" EXCLUDE
)

# generated headers from build dir:
install(FILES
  "${NVBench_BINARY_DIR}/nvbench/detail/version.cuh"
  "${NVBench_BINARY_DIR}/nvbench/detail/git_revision.cuh"

  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/nvbench/detail"
)

# Call with a list of library targets to generate install rules:
function(nvbench_install_libraries)
  install(TARGETS ${ARGN}
    DESTINATION "${NVBench_INSTALL_LIB_DIR}"
    EXPORT nvbench-targets
  )
endfunction()
