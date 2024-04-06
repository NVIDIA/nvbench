# Detect the langauge standards supported by the current compilers.
#
# Usage: detect_supported_cxx_standards(<var_prefix> <lang> <standards>)
#
# - var_prefix: Used to name result variables,
#   e.g. ${var_prefix}_${lang}_XX_SUPPORTED will be TRUE or FALSE. Defined for
#   each XX in ${standards}.
# - lang: The language to test: C, CXX, or CUDA.
# - standards: List of any standard versions.
#
# Example: detect_supported_standards(PROJ CXX 11 14 17)
#   - Sets the following variables in the parent scope to TRUE or FALSE:
#     - PROJ_CXX_11_SUPPORTED
#     - PROJ_CXX_14_SUPPORTED
#     - PROJ_CXX_17_SUPPORTED
#   - Sets `PROJ_DETECTED_CXX_STANDARDS` to a list of supported standards (e.g. "11;14;17").
function(detect_supported_standards prefix lang)
  string(TOLOWER "${lang}_std" feature_prefix)
  set(all_stds)
  foreach(standard IN LISTS ARGN)
    set(var_name "${prefix}_${lang}_${standard}_SUPPORTED")
    if ("${feature_prefix}_${standard}" IN_LIST CMAKE_${lang}_COMPILE_FEATURES)
      set(${var_name} TRUE)
    else()
      set(${var_name} FALSE)
    endif()

    # Special cases:
    if (standard EQUAL 17 AND
        (lang STREQUAL "CXX" OR lang STREQUAL "CUDA") AND
        ((CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
          CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7) OR
         (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND
          CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8)))
      # gcc < 7 and clang < 8 don't fully support C++17.
      # They accept the flag and have partial support, but nvcc will refuse
      # to enable it and falls back to the default dialect for the current
      # CXX compiler version. This breaks our CI.
      # CMake's COMPILE_FEATURES var reports that these compilers support C++17,
      # but we can't rely on it, so manually disable the dialect in these cases.
      set(${var_name} FALSE)
    endif()

    if (standard EQUAL 20 AND
        (lang STREQUAL "CXX" OR lang STREQUAL "CUDA") AND
        ((CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
          CMAKE_CXX_COMPILER_VERSION VERSION_LESS 10) OR
         (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND
          CMAKE_CXX_COMPILER_VERSION VERSION_LESS 10) OR
         (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND
          CMAKE_CXX_COMPILER_VERSION VERSION_LESS 1930)))
      # Similar to the above, but for C++20.
      set(${var_name} FALSE)
    endif()

    if (${var_name})
      list(APPEND all_stds ${standard})
    endif()

    message(STATUS "Testing ${lang}${standard} Support: ${${var_name}}")
    set(${var_name} ${${var_name}} PARENT_SCOPE)
  endforeach()

  set(${prefix}_DETECTED_${lang}_STANDARDS "${all_stds}" PARENT_SCOPE)
endfunction()
