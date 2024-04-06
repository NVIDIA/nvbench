################################################################################
# fmtlib/fmt
include("${rapids-cmake-dir}/cpm/fmt.cmake")

if(NOT BUILD_SHARED_LIBS AND NVBench_ENABLE_INSTALL_RULES)
set(export_set_details BUILD_EXPORT_SET nvbench-targets
                       INSTALL_EXPORT_SET nvbench-targets)
endif()

rapids_cpm_fmt(${export_set_details}
  CPM_ARGS
    OPTIONS
      # Force static to keep fmt internal.
      "BUILD_SHARED_LIBS OFF"
)

if(NOT fmt_ADDED)
  set(fmt_is_external TRUE)
endif()

################################################################################
# nlohmann/json
#
# Following recipe from
# http://github.com/cpm-cmake/CPM.cmake/blob/master/examples/json/CMakeLists.txt
# Download the zips because the repo takes an excessively long time to clone.
rapids_cpm_find(nlohmann_json 3.11.3
  CPM_ARGS
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/include.zip
    URL_HASH SHA256=a22461d13119ac5c78f205d3df1db13403e58ce1bb1794edc9313677313f4a9d
  PATCH_COMMAND
    ${CMAKE_COMMAND}
      -D "CUDA_VERSION=${CMAKE_CUDA_COMPILER_VERSION}"
      -D "CXX_VERSION=${CMAKE_CXX_COMPILER_VERSION}"
      -D "CXX_ID=${CMAKE_CXX_COMPILER_ID}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/json_unordered_map_ice.cmake"
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
