/*
 *  Copyright 2026 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 with the LLVM exception
 *  (the "License"); you may not use this file except in compliance with
 *  the License.
 *
 *  You may obtain a copy of the License at
 *
 *      http://llvm.org/foundation/relicensing/LICENSE.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <nvbench/benchmark.cuh>
#include <nvbench/callable.cuh>
#include <nvbench/json_printer.cuh>
#include <nvbench/state.cuh>
#include <nvbench/summary.cuh>
#include <nvbench/types.cuh>

#include <chrono>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

#include "test_asserts.cuh"

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#error "No <filesystem> or <experimental/filesystem> found."
#endif

void dummy_generator(nvbench::state &) {}
NVBENCH_DEFINE_CALLABLE(dummy_generator, dummy_callable);
using dummy_bench = nvbench::benchmark<dummy_callable>;

struct temp_directory
{
  temp_directory()
      : path{fs::temp_directory_path() /
             ("nvbench-json-printer-test-" +
              std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))}
  {
    fs::create_directories(path);
  }

  ~temp_directory()
  {
    std::error_code ec;
    fs::remove_all(path, ec);
  }

  fs::path path;
};

namespace nvbench::detail
{
// nvbench::state construction is private to the state generator and this test
// hook; use the friend type to build a minimal state for printer testing.
struct state_tester : public nvbench::state
{
  state_tester(const nvbench::benchmark_base &bench)
      : nvbench::state{bench}
  {}
};
} // namespace nvbench::detail

void test_jsonbin_filenames_are_json_relative()
{
  const temp_directory tmp_dir;
  const auto json_path = tmp_dir.path / "result.json";

  std::ofstream json_stream{json_path};
  nvbench::json_printer printer{json_stream, json_path.string(), true};

  dummy_bench bench;
  nvbench::detail::state_tester state{bench};

  printer.process_bulk_data(state, "nv/cold/sample_times", "sample_times", {1.0, 2.0});
  printer.process_bulk_data(state, "nv/cold/sample_freqs", "sample_freqs", {3.0, 4.0});

  const auto samples_filename =
    state.get_summary("nv/json/bin:nv/cold/sample_times").get_string("filename");
  const auto freqs_filename =
    state.get_summary("nv/json/freqs-bin:nv/cold/sample_freqs").get_string("filename");

  ASSERT(fs::path{samples_filename} == fs::path{"result.json-bin"} / "0.bin");
  ASSERT(fs::path{freqs_filename} == fs::path{"result.json-freqs-bin"} / "0.bin");
  ASSERT(fs::exists(tmp_dir.path / samples_filename));
  ASSERT(fs::exists(tmp_dir.path / freqs_filename));
}

int main() { test_jsonbin_filenames_are_json_relative(); }
