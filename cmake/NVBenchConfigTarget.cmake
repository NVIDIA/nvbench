include(CheckCXXCompilerFlag)

option(NVBench_ENABLE_WERROR
  "Treat warnings as errors while compiling NVBench."
  ${NVBench_TOPLEVEL_PROJECT}
)
mark_as_advanced(NVBench_ENABLE_WERROR)

# Builds all NVBench targets (libs, tests, examples, etc).
add_custom_target(nvbench.all)

set(NVBench_LIBRARY_OUTPUT_DIR "${CMAKE_BINARY_DIR}/lib")
set(NVBench_EXECUTABLE_OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin")

add_library(nvbench.build_interface INTERFACE)

# TODO Why must this be installed/exported if it's just a private interface?
# CMake complains about it missing from the export set unless we export it.
# Is there way to avoid this?
set_target_properties(nvbench.build_interface PROPERTIES
  EXPORT_NAME internal_build_interface
)

function(nvbench_add_cxx_flag target_name type flag)
  string(MAKE_C_IDENTIFIER "NVBench_CXX_FLAG_${flag}" var)
  check_cxx_compiler_flag(${flag} ${var})

  if (${${var}})
    target_compile_options(${target_name} ${type}
      $<$<COMPILE_LANGUAGE:CXX>:${flag}>
      $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcompiler=${flag}>
    )
  endif()
endfunction()

# We test to see if C++ compiler options exist using try-compiles in the CXX lang, and then reuse those flags as
# -Xcompiler flags for CUDA targets. This requires that the CXX compiler and CUDA_HOST compilers are the same when
# using nvcc.
if (NVBench_TOPLEVEL_PROJECT AND CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
  set(cuda_host_matches_cxx_compiler FALSE)
  if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.31)
    set(host_info "${CMAKE_CUDA_HOST_COMPILER} (${CMAKE_CUDA_HOST_COMPILER_ID} ${CMAKE_CUDA_HOST_COMPILER_VERSION})")
    set(cxx_info "${CMAKE_CXX_COMPILER} (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})")
    if (CMAKE_CUDA_HOST_COMPILER_ID STREQUAL CMAKE_CXX_COMPILER_ID AND
        CMAKE_CUDA_HOST_COMPILER_VERSION VERSION_EQUAL CMAKE_CXX_COMPILER_VERSION)
      set(cuda_host_matches_cxx_compiler TRUE)
    endif()
  else() # CMake < 3.31 doesn't have the CMAKE_CUDA_HOST_COMPILER_ID/VERSION variables
    set(host_info "${CMAKE_CUDA_HOST_COMPILER}")
    set(cxx_info "${CMAKE_CXX_COMPILER}")
    if (CMAKE_CUDA_HOST_COMPILER STREQUAL CMAKE_CXX_COMPILER)
      set(cuda_host_matches_cxx_compiler TRUE)
    endif()
  endif()

  if (NOT cuda_host_matches_cxx_compiler)
    message(FATAL_ERROR
      "NVBench developer builds require that CMAKE_CUDA_HOST_COMPILER matches "
      "CMAKE_CXX_COMPILER when using nvcc:\n"
      "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}\n"
      "CMAKE_CUDA_HOST_COMPILER: ${host_info}\n"
      "CMAKE_CXX_COMPILER: ${cxx_info}\n"
      "Rerun cmake with \"-DCMAKE_CUDA_HOST_COMPILER=${CMAKE_CXX_COMPILER}\".\n"
      "Alternatively, configure the CUDAHOSTCXX and CXX environment variables to match.\n"
    )
  endif()
endif()

nvbench_add_cxx_flag(nvbench.build_interface INTERFACE "-Wall")
nvbench_add_cxx_flag(nvbench.build_interface INTERFACE "-Wextra")
nvbench_add_cxx_flag(nvbench.build_interface INTERFACE "-Wconversion")
nvbench_add_cxx_flag(nvbench.build_interface INTERFACE "-Woverloaded-virtual")
nvbench_add_cxx_flag(nvbench.build_interface INTERFACE "-Wcast-qual")
nvbench_add_cxx_flag(nvbench.build_interface INTERFACE "-Wpointer-arith")
nvbench_add_cxx_flag(nvbench.build_interface INTERFACE "-Wunused-local-typedef")
nvbench_add_cxx_flag(nvbench.build_interface INTERFACE "-Wunused-parameter")
nvbench_add_cxx_flag(nvbench.build_interface INTERFACE "-Wvla")
nvbench_add_cxx_flag(nvbench.build_interface INTERFACE "-Wgnu")
nvbench_add_cxx_flag(nvbench.build_interface INTERFACE "-Wno-gnu-line-marker") # WAR 3916341

if (NVBench_ENABLE_WERROR)
  nvbench_add_cxx_flag(nvbench.build_interface INTERFACE "-Werror")
endif()

# Experimental filesystem library
if (CMAKE_CXX_COMPILER_ID STREQUAL GNU OR CMAKE_CXX_COMPILER_ID STREQUAL Clang)
  target_link_libraries(nvbench.build_interface INTERFACE stdc++fs)
endif()

# CUDA-specific flags
if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
  # fmtlib uses llvm's _BitInt internally, which is not available when compiling through nvcc:
  target_compile_definitions(nvbench.build_interface INTERFACE "FMT_USE_BITINT=0")
endif()

target_compile_options(nvbench.build_interface INTERFACE
  $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcudafe=--display_error_number>
  $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Wno-deprecated-gpu-targets>
)
if (NVBench_ENABLE_WERROR)
  target_compile_options(nvbench.build_interface INTERFACE
    $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcudafe=--promote_warnings>
  )
endif()

function(nvbench_config_target target_name)
  target_link_libraries(${target_name} PRIVATE nvbench.build_interface)
  set_target_properties(${target_name} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${NVBench_LIBRARY_OUTPUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${NVBench_LIBRARY_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${NVBench_EXECUTABLE_OUTPUT_DIR}"
  )
  # CUPTI libraries are installed in random locations depending on the platform
  # and installation method. Sometimes they're next to the CUDA libraries and in
  # the library path, other times they're in a subdirectory that isn't added to
  # the library path...
  # To simplify installed nvbench usage, add the CUPTI libraries path to the
  # installed nvbench rpath:
  if (NVBench_ENABLE_CUPTI AND nvbench_cupti_root)
    set_target_properties(${target_name} PROPERTIES
      INSTALL_RPATH "${nvbench_cupti_root}/lib64"
    )
  endif()
endfunction()
