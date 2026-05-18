# Since this file is installed, we need to make sure that the CUDAToolkit has
# been found by consumers:
if (NOT TARGET CUDA::toolkit)
  find_package(CUDAToolkit REQUIRED)
endif()

if (EXISTS "${CUDAToolkit_LIBRARY_ROOT}/extras/CUPTI/lib64")
  # NVIDIA installer layout:
  set(nvbench_cupti_root "${CUDAToolkit_LIBRARY_ROOT}/extras/CUPTI")
else()
  # Ubuntu package layout:
  set(nvbench_cupti_root "${CUDAToolkit_LIBRARY_ROOT}")
endif()

set(nvbench_cupti_library_hints "${nvbench_cupti_root}/lib64")
if (WIN32)
  list(APPEND nvbench_cupti_library_hints
    "${nvbench_cupti_root}/lib/x64"
    "${nvbench_cupti_root}/lib"
  )
endif()

# The CUPTI targets in FindCUDAToolkit are broken:
# - The dll locations are not specified
# - Dependent libraries nvperf_* are not linked.
# So we create our own targets:
function(nvbench_find_windows_cupti_runtime_library out_var dep_name library_path)
  cmake_path(GET library_path PARENT_PATH library_dir)
  set(runtime_search_dirs "${library_dir}")

  if ("${library_dir}" MATCHES "/Library/lib/x64$")
    cmake_path(GET library_dir PARENT_PATH conda_lib_dir)
    cmake_path(GET conda_lib_dir PARENT_PATH conda_library_dir)
    list(APPEND runtime_search_dirs "${conda_library_dir}/bin")
  elseif ("${library_dir}" MATCHES "/Library/lib$")
    cmake_path(GET library_dir PARENT_PATH conda_library_dir)
    list(APPEND runtime_search_dirs "${conda_library_dir}/bin")
  endif()

  list(REMOVE_DUPLICATES runtime_search_dirs)

  foreach(runtime_search_dir IN LISTS runtime_search_dirs)
    if ("${dep_name}" STREQUAL "cupti")
      file(GLOB runtime_libraries LIST_DIRECTORIES false
        "${runtime_search_dir}/cupti64_*.dll"
      )
      if (NOT runtime_libraries)
        file(GLOB runtime_libraries LIST_DIRECTORIES false
          "${runtime_search_dir}/cupti.dll"
        )
      endif()
    else()
      file(GLOB runtime_libraries LIST_DIRECTORIES false
        "${runtime_search_dir}/${dep_name}.dll"
      )
    endif()

    if (runtime_libraries)
      list(SORT runtime_libraries COMPARE NATURAL ORDER DESCENDING)
      list(LENGTH runtime_libraries num_runtime_libraries)
      if (num_runtime_libraries GREATER 1)
        list(GET runtime_libraries 0 runtime_library)
        message(WARNING
          "Found multiple runtime DLLs for ${dep_name}; selecting "
          "${runtime_library}. Candidates: ${runtime_libraries}"
        )
      else()
        list(GET runtime_libraries 0 runtime_library)
      endif()

      set(${out_var} "${runtime_library}" PARENT_SCOPE)
      return()
    endif()
  endforeach()

  message(FATAL_ERROR
    "Could not find the runtime DLL for ${dep_name}. "
    "Searched these directories: ${runtime_search_dirs}"
  )
endfunction()

function(nvbench_add_cupti_dep dep_name)
  string(TOLOWER ${dep_name} dep_name_lower)
  string(TOUPPER ${dep_name} dep_name_upper)

  add_library(nvbench::${dep_name_lower} SHARED IMPORTED)

  find_library(NVBench_${dep_name_upper}_LIBRARY ${dep_name_lower} REQUIRED
    DOC "The library for ${dep_name_lower} from the CUDA Toolkit."
    HINTS ${nvbench_cupti_library_hints}
  )
  mark_as_advanced(NVBench_${dep_name_upper}_LIBRARY)

  if (WIN32)
    nvbench_find_windows_cupti_runtime_library(
      NVBench_${dep_name_upper}_DLL
      ${dep_name_lower}
      "${NVBench_${dep_name_upper}_LIBRARY}"
    )
    set_target_properties(nvbench::${dep_name_lower} PROPERTIES
      IMPORTED_IMPLIB "${NVBench_${dep_name_upper}_LIBRARY}"
      IMPORTED_LOCATION "${NVBench_${dep_name_upper}_DLL}"
    )
  else()
    set_target_properties(nvbench::${dep_name_lower} PROPERTIES
      IMPORTED_LOCATION "${NVBench_${dep_name_upper}_LIBRARY}"
    )
  endif()
endfunction()

nvbench_add_cupti_dep(cupti)
target_include_directories(nvbench::cupti INTERFACE
  "${nvbench_cupti_root}/include"
)

if (NOT EXISTS "${nvbench_cupti_root}/include/cupti_profiler_host.h")
  # Profile Host API does not exist yet, need NVPERF libraries
  # for NVPW_* API used in nvbench::cupti_profiler
  nvbench_add_cupti_dep(nvperf_target)
  nvbench_add_cupti_dep(nvperf_host)
  target_link_libraries(nvbench::cupti INTERFACE
    nvbench::nvperf_target
    nvbench::nvperf_host
  )
endif()
