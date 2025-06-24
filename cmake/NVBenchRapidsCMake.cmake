# Called before project(...)
macro(nvbench_load_rapids_cmake)
  # - Including directly, see https://github.com/rapidsai/rmm/pull/1886
  # - Versioned download URL:
  #   https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-XX.YY/RAPIDS.cmake
  # - This macro is always called before project() in the root CMakeLists.txt, so:
  #   - we can't just use NVBench_SOURCE_DIR, it's not defined yet.
  #   - We can't rely on CMAKE_CURRENT_LIST_DIR because of macro expansion.
  #   - We can fallback to CURRENT_SOURCE_DIR because we know this will be expanded in the root:
  include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/RAPIDS.cmake")

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
  rapids_cmake_write_version_file(
    "${NVBench_BINARY_DIR}/nvbench/detail/version.cuh"
    PREFIX "NVBENCH"
  )
  rapids_cpm_init()
endmacro()
