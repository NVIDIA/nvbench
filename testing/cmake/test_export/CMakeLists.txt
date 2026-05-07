cmake_minimum_required(VERSION 3.20.1)
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
get_property(nvbench_config TARGET nvbench::nvbench
  PROPERTY IMPORTED_CONFIGURATIONS
)
list(LENGTH nvbench_config num_configs)
if (num_configs GREATER 1)
  message(WARNING
    "Multiple IMPORTED_CONFIGURATIONS for nvbench::nvbench. "
    "Picking the first one. This may cause issues."
  )
  list(GET nvbench_config 0 nvbench_config)
endif()

set(nvbench_lib_dir "")
# On Unix the build tree uses RUNPATH so only the install tree needs the path.
# On Windows there is no RUNPATH so we always need the DLL directory.
if (WIN32 OR TEST_TYPE STREQUAL "INSTALL_TREE")
  get_property(nvbench_lib TARGET nvbench::nvbench
    PROPERTY IMPORTED_LOCATION_${nvbench_config}
  )
  cmake_path(GET nvbench_lib PARENT_PATH nvbench_lib_dir)
endif()

set(cupti_lib_dir "")
if (TARGET nvbench::cupti)
  if (WIN32)
    get_property(cupti_lib TARGET nvbench::cupti PROPERTY IMPORTED_IMPLIB)
  else()
    get_property(cupti_lib TARGET nvbench::cupti PROPERTY IMPORTED_LOCATION)
  endif()
  cmake_path(GET cupti_lib PARENT_PATH cupti_lib_dir)
endif()

if (WIN32)
  set(lib_dirs "${nvbench_lib_dir}\\;${cupti_lib_dir}")
  set_property(TEST test_bench PROPERTY
    ENVIRONMENT "PATH=${lib_dirs}\\;$ENV{PATH}"
  )
  set_property(TEST nvbench_ctl PROPERTY
    ENVIRONMENT "PATH=${lib_dirs}\\;$ENV{PATH}"
  )
else()
  set_property(TEST test_bench PROPERTY
    ENVIRONMENT "LD_LIBRARY_PATH=${cupti_lib_dir}"
  )
  set_property(TEST nvbench_ctl PROPERTY
    ENVIRONMENT "LD_LIBRARY_PATH=${nvbench_lib_dir}:${cupti_lib_dir}"
  )
endif()
