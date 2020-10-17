include(CPM)

CPMAddPackage(
  NAME benchmark
  GITHUB_REPOSITORY google/benchmark
  VERSION 1.5.2
  OPTIONS
  "BENCHMARK_ENABLE_TESTING Off"
  # Note from cucollections:
  # The REGEX feature test fails when gbench's cmake is run under CPM w/ gcc5.4 because it doesn't assume C++11
  # Additionally, attempting to set the CMAKE_CXX_VERSION here doesn't propogate to the feature test build
  # Therefore, we just disable the feature test and assume platforms we care about have a regex impl available
  "RUN_HAVE_STD_REGEX 0"
)

add_library(nvbench INTERFACE)
target_include_directories(nvbench INTERFACE "${CMAKE_CURRENT_LIST_DIR}")
target_link_libraries(nvbench INTERFACE benchmark_main)
