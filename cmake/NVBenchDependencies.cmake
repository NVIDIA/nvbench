################################################################################
# fmtlib/fmt
rapids_cpm_find(fmt 7.1.3
  CPM_ARGS
    GITHUB_REPOSITORY fmtlib/fmt
    GIT_TAG 7.1.3
    GIT_SHALLOW TRUE
    OPTIONS
      # Force static to keep fmt internal.
      "BUILD_SHARED_LIBS OFF"
      "CMAKE_POSITION_INDEPENDENT_CODE ON"
)

################################################################################
# nlohmann/json
#
# Following recipe from
# http://github.com/cpm-cmake/CPM.cmake/blob/master/examples/json/CMakeLists.txt
# Download the zips because the repo takes an excessively long time to clone.
rapids_cpm_find(nlohmann_json 3.9.1
  # Release:
  CPM_ARGS
    URL https://github.com/nlohmann/json/releases/download/v3.9.1/include.zip
    URL_HASH SHA256=6bea5877b1541d353bd77bdfbdb2696333ae5ed8f9e8cc22df657192218cad91
    PATCH_COMMAND
      # Work around compiler bug in nvcc 11.0, see NVIDIA/NVBench#18
      ${CMAKE_COMMAND} -E copy
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/nlohmann_json.hpp"
        "./include/nlohmann/json.hpp"

  # Development version:
  # I'm waiting for https://github.com/nlohmann/json/issues/2676 to be fixed,
  # leave this in to simplify testing patches as they come out. Update the
  # `nvbench_json` target too when switching branches.
  #  CPM_ARGS
  #    VERSION develop
  #    URL https://github.com/nlohmann/json/archive/refs/heads/develop.zip
  #    OPTIONS JSON_MultipleHeaders ON
)

# nlohmann_json release headers
add_library(nvbench_json INTERFACE IMPORTED)
target_include_directories(nvbench_json SYSTEM INTERFACE
  "${nlohmann_json_SOURCE_DIR}/include"
)

# nlohmann_json development branch:
#add_library(nvbench_json INTERFACE)
#target_link_libraries(nvbench_json INTERFACE nlohmann_json)

################################################################################
# CUDAToolkit
rapids_find_package(CUDAToolkit REQUIRED
  BUILD_EXPORT_SET nvbench-targets
  INSTALL_EXPORT_SET nvbench-targets
)

# Append CTK targets to this as we add optional deps (NMVL, CUPTI, ...)
set(ctk_libraries CUDA::toolkit)

################################################################################
# CUDAToolkit -> NVML
if (NVBench_ENABLE_NVML)
  include("${CMAKE_CURRENT_LIST_DIR}/NVBenchNVML.cmake")
  list(APPEND ctk_libraries nvbench::nvml)
endif()

################################################################################
# CUDAToolkit -> CUPTI
if (NVBench_ENABLE_CUPTI)
  include("${CMAKE_CURRENT_LIST_DIR}/NVBenchCUPTI.cmake")
  list(APPEND ctk_libraries CUDA::cuda_driver nvbench::cupti)
endif()
