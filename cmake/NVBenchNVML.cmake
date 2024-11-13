# Since this file is installed, we need to make sure that the CUDAToolkit has
# been found by consumers:
if (NOT TARGET CUDA::toolkit)
  find_package(CUDAToolkit REQUIRED)
endif()

if (WIN32)
  # The CUDA:: targets currently don't provide dll locations through the
  # `IMPORTED_LOCATION` property, nor are they marked as `SHARED` libraries
  # (they're currently `UNKNOWN`). This prevents the `nvbench_setup_dep_dlls`
  # CMake function from copying the dlls to the build / install directories.
  # This is discussed in https://gitlab.kitware.com/cmake/cmake/-/issues/22845
  # and the other CMake issues it links to.
  #
  # We create a nvbench-specific target that configures the nvml interface as
  # described here:
  # https://gitlab.kitware.com/cmake/cmake/-/issues/22845#note_1077538
  #
  # Use find_file instead of find_library, which would search for a .lib file.
  # This is also nice because find_file searches recursively (find_library
  # does not) and some versions of CTK nest nvml.dll several directories deep
  # under C:\Windows\System32.
  find_file(NVBench_NVML_DLL nvml.dll
    DOC "The full path to nvml.dll. Usually somewhere under C:/Windows/System32."
    PATHS "C:/Windows/System32"
  )
  mark_as_advanced(NVBench_NVML_DLL)
endif()

if (NVBench_NVML_DLL)
  add_library(nvbench::nvml SHARED IMPORTED)
  target_link_libraries(nvbench::nvml INTERFACE CUDA::toolkit)
  set_target_properties(nvbench::nvml PROPERTIES
    IMPORTED_LOCATION "${NVBench_NVML_DLL}"
    IMPORTED_IMPLIB "${CUDA_nvml_LIBRARY}"
  )
elseif(TARGET CUDA::nvml)
  add_library(nvbench::nvml ALIAS CUDA::nvml)
else()
  message(FATAL_ERROR "Could not find nvml.dll or CUDA::nvml target. "
          "Set -DNVBench_ENABLE_NVML=OFF to disable NVML support "
          "or set -DNVBench_NVML_DLL to the full path to nvml.dll on Windows.")
endif()
