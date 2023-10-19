# Called before project(...)
macro(nvbench_load_rapids_cmake)
  if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/NVBENCH_RAPIDS.cmake")
    file(DOWNLOAD
      https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-23.12/RAPIDS.cmake
      "${CMAKE_CURRENT_BINARY_DIR}/NVBENCH_RAPIDS.cmake"
    )
  endif()
  include("${CMAKE_CURRENT_BINARY_DIR}/NVBENCH_RAPIDS.cmake")

  include(rapids-cmake)
  include(rapids-cpm)
  include(rapids-cuda)
  include(rapids-export)
  include(rapids-find)

  rapids_cuda_init_architectures(NVBench)
endmacro()

# Called after project(...)
macro(nvbench_init_rapids_cmake)
  rapids_cmake_build_type(Release)
  rapids_cmake_write_version_file("${NVBench_BINARY_DIR}/nvbench/detail/version.cuh")
  rapids_cpm_init()
endmacro()
