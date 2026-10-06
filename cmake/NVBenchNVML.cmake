# Since this file is installed, we need to make sure that the CUDAToolkit has
# been found by consumers:
if (NOT TARGET CUDA::toolkit)
  find_package(CUDAToolkit REQUIRED)
endif()


if (WIN32)
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

  find_file(NVBench_NVML_DLL nvml.dll
    HINTS ${nvbench_nvml_runtime_hints}
    DOC "The NVML runtime DLL from the NVIDIA driver."
    REQUIRED
  )
  mark_as_advanced(NVBench_NVML_DLL)
endif()
