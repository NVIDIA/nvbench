# For every public header, build a translation unit containing `#include <header>`
# with some various checks.

set(excluded_headers_regexes
  # Should never be used externally.
  "^detail"
  "^internal"
)

# Meta target for all configs' header builds:
add_custom_target(nvbench.headers.all)
add_dependencies(nvbench.all nvbench.headers.all)

file(GLOB_RECURSE header_files
  RELATIVE "${NVBench_SOURCE_DIR}/nvbench/"
  CONFIGURE_DEPENDS
  "${NVBench_SOURCE_DIR}/nvbench/*.cuh"
)

foreach (exclusion IN LISTS excluded_headers_regexes)
  list(FILTER header_files EXCLUDE REGEX "${exclusion}")
endforeach()

function (nvbench_add_header_target target_name cuda_std)
  foreach (header IN LISTS header_files)
    set(headertest_src "headers/${target_name}/${header}.cu")
    set(header_str "nvbench/${header}") # Substitution used by configure_file:
    configure_file("${NVBench_SOURCE_DIR}/cmake/header_test.in.cxx" "${headertest_src}")
    list(APPEND headertest_srcs "${headertest_src}")
  endforeach()

  add_library(${target_name} OBJECT ${headertest_srcs})
  target_link_libraries(${target_name} PUBLIC nvbench::nvbench)
  set_target_properties(${target_name} PROPERTIES COMPILE_FEATURES cuda_std_${cuda_std})
  add_dependencies(nvbench.headers.all ${target_name})
endfunction()

foreach (std IN LISTS NVBench_DETECTED_CUDA_STANDARDS)
  nvbench_add_header_target(nvbench.headers.cpp${std} ${std})
endforeach()
