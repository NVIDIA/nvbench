include(CheckCompilerFlag)

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
  string(MAKE_C_IDENTIFIER "NVBench_CXX_FLAG_${flag}" cxx_var)
  check_compiler_flag(CXX "${flag}" ${cxx_var})

  if (${cxx_var})
    target_compile_options(${target_name} ${type}
      $<$<COMPILE_LANGUAGE:CXX>:${flag}>
    )
  endif()

  if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    string(MAKE_C_IDENTIFIER "NVBench_CUDA_HOST_FLAG_${flag}" cuda_host_var)
    check_compiler_flag(CUDA "-Xcompiler=${flag}" ${cuda_host_var})

    if (${cuda_host_var})
      target_compile_options(${target_name} ${type}
        $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcompiler=${flag}>
      )
    endif()
  endif()
endfunction()

# Check warning support independently for the C++ compiler and the CUDA host compiler. The top-level developer
# build check below still requires matching compilers when using nvcc.
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
nvbench_add_cxx_flag(nvbench.build_interface INTERFACE "-Wunused-local-typedefs")
nvbench_add_cxx_flag(nvbench.build_interface INTERFACE "-Wunused-parameter")
nvbench_add_cxx_flag(nvbench.build_interface INTERFACE "-Wvla")
nvbench_add_cxx_flag(nvbench.build_interface INTERFACE "-Wgnu")
nvbench_add_cxx_flag(nvbench.build_interface INTERFACE "-Wno-gnu-line-marker") # WAR 3916341

if (NVBench_ENABLE_WERROR)
  nvbench_add_cxx_flag(nvbench.build_interface INTERFACE "-Werror")
endif()

# Match the filesystem header selection used by the CUDA sources.
block()
  set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
  set(filesystem_source [=[
#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#error "No <filesystem> or <experimental/filesystem> found."
#endif

bool filesystem_exists(const char *path) { return fs::exists(path); }
]=])
  set(main_source [=[
bool filesystem_exists(const char *path);
int main(int argc, char **argv) { return argc > 1 && filesystem_exists(argv[1]); }
]=])

  # C++ and CUDA may use different host compilers and standard libraries.
  # Probe both independently, retaining the C++ linker used by NVBench.
  set(filesystem_sources filesystem.cxx)
  get_property(enabled_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
  if ("CUDA" IN_LIST enabled_languages)
    list(APPEND filesystem_sources filesystem.cu)
  endif()

  # Prefer the matching filesystem archive when available. GCC 8 headers can
  # otherwise link to incompatible symbols in a newer libstdc++.so and crash.
  set(filesystem_needs_stdcxxfs FALSE)
  foreach(filesystem_source_name IN LISTS filesystem_sources)
    try_compile(filesystem_links
      SOURCE_FROM_VAR ${filesystem_source_name} filesystem_source
      SOURCE_FROM_VAR main.cxx main_source
      LINKER_LANGUAGE CXX
      LINK_LIBRARIES stdc++fs
      NO_CACHE
    )

    if (filesystem_links)
      set(filesystem_needs_stdcxxfs TRUE)
    else()
      try_compile(filesystem_links
        SOURCE_FROM_VAR ${filesystem_source_name} filesystem_source
        SOURCE_FROM_VAR main.cxx main_source
        LINKER_LANGUAGE CXX
        NO_CACHE
      )
      if (NOT filesystem_links)
        message(FATAL_ERROR
          "NVBench could not link ${filesystem_source_name} with or without stdc++fs. "
          "See CMakeConfigureLog.yaml."
        )
      endif()
    endif()
  endforeach()

  if (filesystem_needs_stdcxxfs)
    target_link_libraries(nvbench.build_interface INTERFACE stdc++fs)
  endif()
endblock()

# CUDA-specific flags
if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
  # fmtlib uses llvm's _BitInt internally, which is not available when compiling through nvcc:
  target_compile_definitions(nvbench.build_interface INTERFACE "FMT_USE_BITINT=0")
  if (MSVC)
    # cudafe cannot evaluate fmtlib's UTF-8 literal check even when /utf-8 is passed to the host compiler:
    target_compile_definitions(nvbench.build_interface INTERFACE
      $<$<COMPILE_LANGUAGE:CUDA>:FMT_UNICODE=0>
    )
  endif()
endif()

target_compile_options(nvbench.build_interface INTERFACE
  $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcudafe=--display_error_number>
  $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Wno-deprecated-gpu-targets>
  $<$<AND:$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/utf-8>
  # Suppress cudafe diagnostics triggered by fmtlib headers when compiled through MSVC+nvcc:
  #   27: character value is out of range (char32_t sentinel values in lookup tables)
  #  128: loop is not reachable (dead code in constexpr string comparison)
  # 2417: constexpr constructor calls non-constexpr function (bigint default ctor)
  $<$<AND:$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>,$<CXX_COMPILER_ID:MSVC>>:-Xcudafe=--diag_suppress=27>
  $<$<AND:$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>,$<CXX_COMPILER_ID:MSVC>>:-Xcudafe=--diag_suppress=128>
  $<$<AND:$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>,$<CXX_COMPILER_ID:MSVC>>:-Xcudafe=--diag_suppress=2417>
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
  # Final executables need their own copy step when nvbench is static or the
  # control executable is disabled/not built. The helper skips static targets.
  nvbench_setup_dep_dlls(${target_name})

  # CUPTI libraries are installed in random locations depending on the platform
  # and installation method. Sometimes they're next to the CUDA libraries and in
  # the library path, other times they're in a subdirectory that isn't added to
  # the library path...
  # To simplify installed nvbench usage, add the CUPTI libraries path to the
  # installed nvbench rpath (Unix only; Windows uses PATH for DLL lookup):
  if (NVBench_ENABLE_CUPTI AND nvbench_cupti_root AND NOT WIN32)
    set_target_properties(${target_name} PROPERTIES
      INSTALL_RPATH "${nvbench_cupti_root}/lib64"
    )
  endif()
endfunction()

function(nvbench_append_test_runtime_path path_modifications_var target_name)
  if (NOT TARGET ${target_name})
    return()
  endif()

  list(APPEND ${path_modifications_var}
    "PATH=path_list_prepend:$<TARGET_FILE_DIR:${target_name}>"
  )

  set(${path_modifications_var}
    "${${path_modifications_var}}"
    PARENT_SCOPE
  )
endfunction()

function(nvbench_config_test_runtime_environment test_name)
  if (NOT WIN32)
    return()
  endif()

  set(path_modifications "")
  if (TARGET nvbench)
    nvbench_append_test_runtime_path(path_modifications nvbench)
  else()
    nvbench_append_test_runtime_path(path_modifications nvbench::nvbench)
  endif()

  nvbench_append_test_runtime_path(path_modifications nvbench::cupti)
  nvbench_append_test_runtime_path(path_modifications nvbench::nvperf_target)
  nvbench_append_test_runtime_path(path_modifications nvbench::nvperf_host)

  if (path_modifications)
    list(REMOVE_DUPLICATES path_modifications)
    set_property(TEST ${test_name}
      APPEND PROPERTY ENVIRONMENT_MODIFICATION ${path_modifications}
    )
  endif()
endfunction()
