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

# The CUPTI targets in FindCUDAToolkit are broken:
# - The dll locations are not specified
# - Dependent libraries nvperf_* are not linked.
# So we create our own targets:
function(nvbench_add_cupti_dep dep_name)
  string(TOLOWER ${dep_name} dep_name_lower)
  string(TOUPPER ${dep_name} dep_name_upper)

  add_library(nvbench::${dep_name_lower} SHARED IMPORTED)

  if (WIN32)
    # Attempt to locate the dll in the expected location. This is necessary
    # because the CUPTI dll has a versioned suffix, so we can't directly search
    # for it with find_file.
    file(GLOB dep_dll_path "${nvbench_cupti_root}/lib64/${dep_name_lower}*dll")
    cmake_path(GET dep_dll_path FILENAME dep_dll_filename)

    # If the dll was not found in the expected location, use a default filename as a user hint.
    if (NOT dep_dll_filename)
      set(dep_dll_filename ${dep_name_lower}.dll)
    endif()

    # Use find_file to create a cache variable and mark the file as REQUIRED.
    find_file(NVBench_${dep_name_upper}_DLL ${dep_dll_filename} REQUIRED
      DOC "The full path to ${dep_name_lower}.dll from the CUDA Toolkit."
      HINTS "${nvbench_cupti_root}/lib64/"
    )
    mark_as_advanced(NVBench_${dep_name_upper}_DLL)

    # The .libs don't have suffixes, so we can just directly search for them.
    find_library(NVBench_${dep_name_upper}_LIBRARY ${dep_name_lower}.lib REQUIRED
      DOC "The full path to ${dep_name_lower}.lib from the CUDA Toolkit."
      HINTS "${nvbench_cupti_root}/lib64/"
    )
    mark_as_advanced(NVBench_${dep_name_upper}_LIBRARY)

    set_target_properties(nvbench::${dep_name_lower} PROPERTIES
      IMPORTED_LOCATION "${NVBench_${dep_name_upper}_DLL}"
      IMPORTED_IMPLIB "${NVBench_${dep_name_upper}_LIBRARY}"
    )
  else()
    find_library(NVBench_${dep_name_upper}_LIBRARY ${dep_name_lower} REQUIRED
      DOC "The full path to lib${dep_name_lower}.so from the CUDA Toolkit."
      HINTS "${nvbench_cupti_root}/lib64"
    )
    mark_as_advanced(NVBench_${dep_name_upper}_LIBRARY)

    set_target_properties(nvbench::${dep_name_lower} PROPERTIES
      IMPORTED_LOCATION "${NVBench_${dep_name_upper}_LIBRARY}"
    )
  endif()
endfunction()

nvbench_add_cupti_dep(nvperf_target)
nvbench_add_cupti_dep(nvperf_host)
nvbench_add_cupti_dep(cupti)
target_link_libraries(nvbench::cupti INTERFACE
  nvbench::nvperf_target
  nvbench::nvperf_host
)
target_include_directories(nvbench::cupti INTERFACE
  "${nvbench_cupti_root}/include"
)
