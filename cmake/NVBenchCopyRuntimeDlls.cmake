if (NOT DEFINED NVBENCH_RUNTIME_DLLS_FILE OR
    NOT DEFINED NVBENCH_RUNTIME_DLLS_DEST OR
    NOT DEFINED NVBENCH_RUNTIME_DLLS_LOCK)
  message(FATAL_ERROR "Runtime DLL manifest, destination, and lock are required")
endif()

if (NOT EXISTS "${NVBENCH_RUNTIME_DLLS_FILE}")
  message(FATAL_ERROR "Runtime DLL manifest does not exist: ${NVBENCH_RUNTIME_DLLS_FILE}")
endif()

file(LOCK "${NVBENCH_RUNTIME_DLLS_LOCK}" GUARD PROCESS TIMEOUT 120)

file(READ "${NVBENCH_RUNTIME_DLLS_FILE}" runtime_dlls)
string(REPLACE ";" "\n" runtime_dlls "${runtime_dlls}")
string(REPLACE "\n" ";" runtime_dlls "${runtime_dlls}")

foreach(runtime_dll IN LISTS runtime_dlls)
  if (runtime_dll STREQUAL "")
    continue()
  endif()

  if (NOT EXISTS "${runtime_dll}")
    message(FATAL_ERROR "Runtime DLL does not exist: ${runtime_dll}")
  endif()

  get_filename_component(runtime_dll_name "${runtime_dll}" NAME)
  set(destination_file "${NVBENCH_RUNTIME_DLLS_DEST}/${runtime_dll_name}")
  if (EXISTS "${destination_file}")
    file(REAL_PATH "${runtime_dll}" runtime_dll_real_path)
    file(REAL_PATH "${destination_file}" destination_real_path)
    if (runtime_dll_real_path STREQUAL destination_real_path)
      continue()
    endif()
  endif()

  file(COPY_FILE "${runtime_dll}" "${destination_file}" ONLY_IF_DIFFERENT)
endforeach()
