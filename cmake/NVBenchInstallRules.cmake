
if(NVBench_ENABLE_INSTALL_RULES)

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
  install(
    FILES
      "${NVBench_BINARY_DIR}/nvbench/config.cuh"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/nvbench"
  )
  install(
    FILES
      "${NVBench_BINARY_DIR}/nvbench/detail/version.cuh"
      "${NVBench_BINARY_DIR}/nvbench/detail/git_revision.cuh"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/nvbench/detail"
  )

  #
  # Install CMake files needed by consumers to locate dependencies:
  #

  # Borrowing this logic from rapids_cmake's export logic to make sure these end
  # up in the same location as nvbench-config.cmake:
  rapids_cmake_install_lib_dir(config_install_location)
  set(config_install_location "${config_install_location}/cmake/nvbench")

  if (NVBench_ENABLE_NVML)
    install(
      FILES
        "${NVBench_SOURCE_DIR}/cmake/NVBenchNVML.cmake"
      DESTINATION "${config_install_location}"
    )
  endif()

  if (NVBench_ENABLE_CUPTI)
    install(
      FILES
        "${NVBench_SOURCE_DIR}/cmake/NVBenchCUPTI.cmake"
      DESTINATION "${config_install_location}"
    )
  endif()
endif()

# Call with a list of library targets to generate install rules:
function(nvbench_install_libraries)
  if(NVBench_ENABLE_INSTALL_RULES)
    install(TARGETS ${ARGN}
      DESTINATION "${NVBench_INSTALL_LIB_DIR}"
      EXPORT nvbench-targets
    )
  endif()
endfunction()

# Call with a list of executables to generate install rules:
function(nvbench_install_executables)
  if(NVBench_ENABLE_INSTALL_RULES)
    install(TARGETS ${ARGN} EXPORT nvbench-targets)
  endif()
endfunction()
