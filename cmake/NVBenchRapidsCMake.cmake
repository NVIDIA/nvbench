# Called before project(...)
macro(nvbench_load_rapids_cmake)
  file(DOWNLOAD
    https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-21.12/RAPIDS.cmake
    "${CMAKE_BINARY_DIR}/RAPIDS.cmake"
  )
  include("${CMAKE_BINARY_DIR}/RAPIDS.cmake")

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
  rapids_cmake_write_git_revision_file(
    nvbench_git_revision
    "${NVBench_BINARY_DIR}/nvbench/detail/git_revision.cuh"
  )
  rapids_cpm_init()
endmacro()
