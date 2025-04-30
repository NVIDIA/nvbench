/*
 *  Copyright 2025 NVIDIA Corporation
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

#include <nvbench/nvbench.cuh>

// Grab some testing kernels from NVBench:
#include <nvbench/test_kernels.cuh>

// #define PRINT_DEFAULT_SUMMARY_TAGS

void summary_example(nvbench::state &state)
{
  // Fetch parameters and compute duration in seconds:
  const auto ms       = static_cast<nvbench::float64_t>(state.get_int64("ms"));
  const auto us       = static_cast<nvbench::float64_t>(state.get_int64("us"));
  const auto duration = ms * 1e-3 + us * 1e-6;

  // Add a new column to the summary table with the derived duration used by the benchmark.
  // See the documentation in nvbench/summary.cuh for more details.
  {
    nvbench::summary &summary = state.add_summary("duration");
    summary.set_string("name", "Duration (s)");
    summary.set_string("description", "The duration of the kernel execution.");
    summary.set_string("hint", "duration");
    summary.set_float64("value", duration);
  }

  // Run the measurements:
  state.exec(nvbench::exec_tag::no_batch, [duration](nvbench::launch &launch) {
    nvbench::sleep_kernel<<<1, 1, 0, launch.get_stream()>>>(duration);
  });

#ifdef PRINT_DEFAULT_SUMMARY_TAGS
  // The default summary tags can be found by inspecting the state after calling
  // state.exec.
  // They can also be found by looking at the json output (--json <filename>)
  for (const auto &summary : state.get_summaries())
  {
    std::cout << summary.get_tag() << std::endl;
  }
#endif

  // Default summary columns can be shown/hidden in the markdown output tables by adding/removing
  // the "hide" key. Modify this benchmark to show the minimum and maximum GPUs times, but hide the
  // mean GPU time and all CPU times. SM Clock frequency and throttling info are also shown.
  state.get_summary("nv/cold/time/gpu/min").remove_value("hide");
  state.get_summary("nv/cold/time/gpu/max").remove_value("hide");
  state.get_summary("nv/cold/time/gpu/mean").set_string("hide", "");
  state.get_summary("nv/cold/time/cpu/mean").set_string("hide", "");
  state.get_summary("nv/cold/time/cpu/min").set_string("hide", "");
  state.get_summary("nv/cold/time/cpu/max").set_string("hide", "");
  state.get_summary("nv/cold/time/cpu/stdev/relative").set_string("hide", "");
  state.get_summary("nv/cold/sm_clock_rate/mean").remove_value("hide");
  state.get_summary("nv/cold/sm_clock_rate/scaling/percent").remove_value("hide");
}
NVBENCH_BENCH(summary_example)
  .add_int64_axis("ms", nvbench::range(10, 50, 20))
  .add_int64_axis("us", nvbench::range(100, 500, 200));
