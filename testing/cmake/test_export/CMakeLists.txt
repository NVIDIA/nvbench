cmake_minimum_required(VERSION 3.30.4)
project(NVBenchTestExport CUDA CXX)

message(STATUS "NVBench_DIR=${NVBench_DIR}")
find_package(NVBench REQUIRED)

add_executable(test_bench test_bench.cu)
target_link_libraries(test_bench PRIVATE nvbench::main)
enable_testing()
add_test(NAME test_bench COMMAND "$<TARGET_FILE:test_bench>" --timeout 1)
add_test(NAME nvbench_ctl COMMAND "$<TARGET_FILE:nvbench::ctl>")

# Setup runtime library paths for testing.
# Unix uses LD_LIBRARY_PATH; Windows uses PATH for DLL lookup.
function(get_imported_location out_var target_name)
  get_property(imported_configs TARGET ${target_name}
    PROPERTY IMPORTED_CONFIGURATIONS
  )
  list(LENGTH imported_configs num_configs)
  if (num_configs GREATER 1)
    message(WARNING
      "Multiple IMPORTED_CONFIGURATIONS for ${target_name}. "
      "Picking the first one. This may cause issues."
    )
  endif()

  if (num_configs GREATER 0)
    list(GET imported_configs 0 imported_config)
    get_property(imported_location TARGET ${target_name}
      PROPERTY IMPORTED_LOCATION_${imported_config}
    )
  else()
    get_property(imported_location TARGET ${target_name}
      PROPERTY IMPORTED_LOCATION
    )
  endif()

  set(${out_var} "${imported_location}" PARENT_SCOPE)
endfunction()

set(nvbench_lib_dir "")
# On Unix the build tree uses RUNPATH so only the install tree needs the path.
# On Windows there is no RUNPATH so we always need the DLL directory.
if (WIN32 OR TEST_TYPE STREQUAL "INSTALL_TREE")
  get_imported_location(nvbench_lib nvbench::nvbench)
  if (nvbench_lib)
    cmake_path(GET nvbench_lib PARENT_PATH nvbench_lib_dir)
  endif()
endif()

set(cupti_lib_dir "")
if (TARGET nvbench::cupti)
  get_imported_location(cupti_lib nvbench::cupti)
  if (cupti_lib)
    cmake_path(GET cupti_lib PARENT_PATH cupti_lib_dir)
  endif()
endif()

function(join_nonempty_paths out_var separator)
  set(paths "")
  foreach(path IN LISTS ARGN)
    if (path)
      list(APPEND paths "${path}")
    endif()
  endforeach()
  list(JOIN paths "${separator}" joined_paths)
  set(${out_var} "${joined_paths}" PARENT_SCOPE)
endfunction()

function(prepend_runtime_path out_var path_var separator env_var)
  if (DEFINED ENV{${env_var}} AND NOT "$ENV{${env_var}}" STREQUAL "")
    set(${out_var} "${${path_var}}${separator}$ENV{${env_var}}" PARENT_SCOPE)
  else()
    set(${out_var} "${${path_var}}" PARENT_SCOPE)
  endif()
endfunction()

if (WIN32)
  set(path_modifications "")
  if (cupti_lib_dir)
    list(APPEND path_modifications "PATH=path_list_prepend:$<SHELL_PATH:${cupti_lib_dir}>")
  endif()
  if (nvbench_lib_dir)
    list(APPEND path_modifications "PATH=path_list_prepend:$<SHELL_PATH:${nvbench_lib_dir}>")
  endif()
  if (path_modifications)
    set_property(TEST test_bench PROPERTY
      ENVIRONMENT_MODIFICATION ${path_modifications}
    )
    set_property(TEST nvbench_ctl PROPERTY
      ENVIRONMENT_MODIFICATION ${path_modifications}
    )
  endif()
else()
  join_nonempty_paths(cupti_ld_library_path ":" "${cupti_lib_dir}")
  join_nonempty_paths(nvbench_ld_library_path ":" "${nvbench_lib_dir}" "${cupti_lib_dir}")
  if (cupti_ld_library_path)
    prepend_runtime_path(test_bench_ld_library_path cupti_ld_library_path ":" LD_LIBRARY_PATH)
    set_property(TEST test_bench PROPERTY
      ENVIRONMENT "LD_LIBRARY_PATH=${test_bench_ld_library_path}"
    )
  endif()
  if (nvbench_ld_library_path)
    prepend_runtime_path(nvbench_ctl_ld_library_path nvbench_ld_library_path ":" LD_LIBRARY_PATH)
    set_property(TEST nvbench_ctl PROPERTY
      ENVIRONMENT "LD_LIBRARY_PATH=${nvbench_ctl_ld_library_path}"
    )
  endif()
endif()
