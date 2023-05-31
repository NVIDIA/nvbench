################################################################################
# fmtlib/fmt
rapids_cpm_find(fmt 9.1.0
  CPM_ARGS
    GITHUB_REPOSITORY fmtlib/fmt
    GIT_TAG 9.1.0
    GIT_SHALLOW TRUE
    OPTIONS
      # Force static to keep fmt internal.
      "BUILD_SHARED_LIBS OFF"
      "CMAKE_POSITION_INDEPENDENT_CODE ON"
)
if(NOT fmt_ADDED)
  set(fmt_is_external TRUE)
endif()

if(TARGET fmt::fmt AND NOT TARGET fmt)
  add_library(fmt ALIAS fmt::fmt)
endif()

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
  # leave this in to simplify testing patches as they come out.
  #  CPM_ARGS
  #    VERSION develop
  #    URL https://github.com/nlohmann/json/archive/refs/heads/develop.zip
  #    OPTIONS JSON_MultipleHeaders ON
)

add_library(nvbench_json INTERFACE IMPORTED)
if (TARGET nlohmann_json::nlohmann_json)
  # If we have a target, just use it. Cannot be an ALIAS library because
  # nlohmann_json::nlohmann_json itself might be one.
  target_link_libraries(nvbench_json INTERFACE nlohmann_json::nlohmann_json)
else()
  # Otherwise we only downloaded the headers.
  target_include_directories(nvbench_json SYSTEM INTERFACE
    "${nlohmann_json_SOURCE_DIR}/include"
  )
endif()

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
