# NVCC 11.1 and GCC 9 need a patch to build, otherwise:
#
# nlohmann/ordered_map.hpp(29): error #3316:
# Internal Compiler Error (codegen): "internal error during structure layout!"
#
# Usage:
# ${CMAKE_COMMAND}
#   -D "CUDA_VERSION=${CMAKE_CUDA_COMPILER_VERSION}"
#   -D "CXX_VERSION=${CMAKE_CXX_COMPILER_VERSION}"
#   -D "CXX_ID=${CMAKE_CXX_COMPILER_ID}"
#   -P "json_unordered_map_ice.cmake"

if(CUDA_VERSION VERSION_GREATER 11.8 OR NOT CXX_ID STREQUAL "GNU" OR CXX_VERSION VERSION_LESS 9.0)
  return()
endif()

# Read the file and replace the string "JSON_NO_UNIQUE_ADDRESS" with
# "/* JSON_NO_UNIQUE_ADDRESS */".
file(READ "include/nlohmann/ordered_map.hpp" NLOHMANN_ORDERED_MAP_HPP)
string(REPLACE "JSON_NO_UNIQUE_ADDRESS" "/* [NVBench Patch] JSON_NO_UNIQUE_ADDRESS */"
  NLOHMANN_ORDERED_MAP_HPP "${NLOHMANN_ORDERED_MAP_HPP}")
file(WRITE "include/nlohmann/ordered_map.hpp" "${NLOHMANN_ORDERED_MAP_HPP}")
