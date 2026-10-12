// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <nvbench/benchmark.cuh>
#include <nvbench/create.cuh>
#include <nvbench/cuda_stream.cuh>
#include <nvbench/cupti_timer.cuh>
#include <nvbench/option_parser.cuh>
#include <nvbench/state.cuh>
#include <nvbench/test_kernels.cuh>
#include <nvbench/type_list.cuh>

#include <fmt/format.h>

#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include "test_asserts.cuh"

void DummyBench(nvbench::state &state) { state.skip("Skipping for testing."); }

NVBENCH_BENCH(DummyBench).clear_devices();

namespace
{

constexpr double kernel_time = 0.002;

void test_excludes_host_gap()
{
  nvbench::cuda_stream stream;
  nvbench::cupti_timer timer;

  timer.start(stream);
  nvbench::sleep_kernel<<<1, 1, 0, stream>>>(kernel_time);
  NVBENCH_CUDA_CALL(cudaStreamSynchronize(stream));
  std::this_thread::sleep_for(std::chrono::milliseconds{20});
  nvbench::sleep_kernel<<<1, 1, 0, stream>>>(kernel_time);
  timer.stop(stream);

  const auto duration = timer.get_duration();
  ASSERT_MSG(duration >= 2 * kernel_time * 0.9 && duration < 0.010,
             "Expected ~{}s of busy time excluding the 20ms host gap, got {}s",
             2 * kernel_time,
             duration);
  ASSERT(timer.get_duration() == duration);
}

void test_excludes_work_outside_window()
{
  nvbench::cuda_stream stream;
  nvbench::cupti_timer timer;

  nvbench::sleep_kernel<<<1, 1, 0, stream>>>(0.05);
  timer.start(stream);
  nvbench::sleep_kernel<<<1, 1, 0, stream>>>(kernel_time);
  timer.stop(stream);
  nvbench::sleep_kernel<<<1, 1, 0, stream>>>(0.05);

  const auto duration = timer.get_duration();
  ASSERT_MSG(duration >= kernel_time * 0.9 && duration < 0.010,
             "Expected ~{}s, work outside the window leaked in: {}s",
             kernel_time,
             duration);
}

void test_concurrent_kernels_not_double_counted()
{
  nvbench::cuda_stream stream1;
  nvbench::cuda_stream stream2;
  nvbench::cupti_timer timer;

  timer.start(stream1);
  nvbench::sleep_kernel<<<1, 1, 0, stream1>>>(0.02);
  nvbench::sleep_kernel<<<1, 1, 0, stream2>>>(0.02);
  timer.stop(stream1);

  const auto duration = timer.get_duration();
  ASSERT_MSG(duration >= 0.02 * 0.9 && duration < 0.035,
             "Expected ~0.02s for two overlapping 0.02s kernels, got {}s",
             duration);
}

struct sleep_generator
{
  void operator()(nvbench::state &state, nvbench::type_list<>) const
  {
    state.exec([](nvbench::launch &launch) {
      nvbench::sleep_kernel<<<1, 1, 0, launch.get_stream()>>>(1e-4);
    });
  }
};

void test_measurements(bool run_once, const char *expected)
{
  nvbench::benchmark<sleep_generator> bench;
  bench.add_device(0);
  bench.set_cupti_timer(true);
  bench.set_run_once(run_once);
  bench.set_min_samples(5);
  bench.set_timeout(1.0);
  bench.set_batch_target_time(0.01);
  bench.run();

  const auto &states = bench.get_states();
  ASSERT(states.size() == 1);
  const auto &state = states.front();
  ASSERT_MSG(!state.is_skipped(), "Skipped: {}", state.get_skip_reason());

  std::vector<std::string> tags{"nv/cold/time/gpu/mean"};
  if (!run_once)
  {
    tags.push_back("nv/batch/time/gpu/mean");
  }
  for (const auto &tag : tags)
  {
    const auto &summ = state.get_summary(tag);
    const auto desc  = summ.get_string("description");
    const auto value = summ.get_float64("value");
    ASSERT_MSG(desc.find(expected) != std::string::npos, "{}: '{}'", tag, desc);
    ASSERT_MSG(value >= 1e-4 * 0.9 && value < 1e-3, "{}: {}s", tag, value);
  }
}

// Syncing inside the launcher without exec_tag::sync deadlocks behind a blocking kernel.
struct syncing_generator
{
  void operator()(nvbench::state &state, nvbench::type_list<>) const
  {
    state.set_blocking_kernel_timeout(1.0);
    state.exec([](nvbench::launch &launch) {
      nvbench::sleep_kernel<<<1, 1, 0, launch.get_stream()>>>(1e-4);
      NVBENCH_CUDA_CALL(cudaStreamSynchronize(launch.get_stream()));
    });
  }
};

void test_no_blocking_kernel()
{
  nvbench::benchmark<syncing_generator> bench;
  bench.add_device(0);
  bench.set_cupti_timer(true);
  bench.set_min_samples(5);
  bench.set_timeout(1.0);
  bench.set_batch_target_time(0.01);
  bench.run();

  const auto &state = bench.get_states().front();
  ASSERT_MSG(!state.is_skipped(), "Skipped: {}", state.get_skip_reason());
  ASSERT(state.get_summary("nv/cold/time/gpu/mean").get_float64("value") > 0.);
  ASSERT(state.get_summary("nv/batch/time/gpu/mean").get_float64("value") > 0.);
}

void test_option_parser()
{
  nvbench::option_parser parser;
  parser.parse({"--cupti-timer"});
  for (const auto &bench : parser.get_benchmarks())
  {
    ASSERT(bench->get_cupti_timer());
  }
}

} // namespace

int main()
{
  test_excludes_host_gap();
  test_excludes_work_outside_window();
  test_concurrent_kernels_not_double_counted();
  test_measurements(false, "measured with CUPTI");
  test_measurements(true, "measured with CUDA events");
  test_no_blocking_kernel();
  test_option_parser();
}
