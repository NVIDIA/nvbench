# Since this file is installed, we need to make sure that the CUDAToolkit has
# been found by consumers:
if (NOT TARGET CUDA::toolkit)
  find_package(CUDAToolkit REQUIRED)
endif()

if (NOT TARGET CUDA::nvml)
  message(FATAL_ERROR "NVML support is enabled, but CUDA::nvml was not found.")
endif()
if (NOT TARGET nvbench::nvml)
  add_library(nvbench::nvml ALIAS CUDA::nvml)
endif()

if (WIN32 AND NVBench_ADD_DEPENDENT_DLLS_TO_BUILD)
  # FindCUDAToolkit exposes nvml as an UNKNOWN imported target, so
  # TARGET_RUNTIME_DLLS cannot discover its runtime DLL.
  set(nvbench_nvml_runtime_hints)
  if (DEFINED ENV{ProgramW6432})
    list(APPEND nvbench_nvml_runtime_hints
      "$ENV{ProgramW6432}/NVIDIA Corporation/NVSMI"
    )
  endif()
  if (DEFINED ENV{WINDIR})
    list(APPEND nvbench_nvml_runtime_hints "$ENV{WINDIR}/System32")
  endif()

  # Compile-only Windows runners may not have the NVIDIA driver installed.
  # Copy the NVML runtime when available, but do not require it to configure.
  find_file(NVBench_NVML_DLL nvml.dll
    HINTS ${nvbench_nvml_runtime_hints}
    DOC "The NVML runtime DLL from the NVIDIA driver."
    NO_DEFAULT_PATH
  )
  mark_as_advanced(NVBench_NVML_DLL)
endif()
