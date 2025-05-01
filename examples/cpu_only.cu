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

#include <chrono>
#include <thread>

// Block execution of the current CPU thread for `seconds` seconds.
void sleep_host(double seconds)
{
  std::this_thread::sleep_for(
    std::chrono::milliseconds(static_cast<nvbench::int64_t>(seconds * 1000)));
}

//=============================================================================
// Simple CPU-only benchmark that sleeps on host for a specified duration.
void simple(nvbench::state &state)
{
  const auto duration = state.get_float64("Duration");

  state.exec([duration](nvbench::launch &) { sleep_host(duration); });
}
NVBENCH_BENCH(simple)
  // 100 -> 500 ms in 100 ms increments.
  .add_float64_axis("Duration", nvbench::range(.1, .5, .1))
  // Mark as CPU-only.
  .set_is_cpu_only(true);

//=============================================================================
// Simple CPU-only benchmark that sleeps on host for a specified duration and
// uses a custom timed region.
void simple_timer(nvbench::state &state)
{
  const auto duration = state.get_float64("Duration");

  state.exec(nvbench::exec_tag::timer, [duration](nvbench::launch &, auto &timer) {
    // Do any setup work before starting the timer here...
    timer.start();

    // The region of code to be timed:
    sleep_host(duration);

    timer.stop();
    // Any per-run cleanup here...
  });
}
NVBENCH_BENCH(simple_timer)
  // 100 -> 500 ms in 100 ms increments.
  .add_float64_axis("Duration", nvbench::range(.1, .5, .1))
  // Mark as CPU-only.
  .set_is_cpu_only(true);

//=============================================================================
// Simple CPU-only benchmark that uses the optional `nvbench::exec_tag::no_gpu`
// hint to prevent GPU measurement code from being instantiated. Note that
// `set_is_cpu_only(true)` is still required when using this hint.
void simple_no_gpu(nvbench::state &state)
{
  const auto duration = state.get_float64("Duration");

  state.exec(nvbench::exec_tag::no_gpu, [duration](nvbench::launch &) { sleep_host(duration); });
}
NVBENCH_BENCH(simple_no_gpu)
  // 100 -> 500 ms in 100 ms increments.
  .add_float64_axis("Duration", nvbench::range(.1, .5, .1))
  // Mark as CPU-only.
  .set_is_cpu_only(true);
