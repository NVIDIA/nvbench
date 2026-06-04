// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <nvbench/benchmark.cuh>
#include <nvbench/callable.cuh>
#include <nvbench/criterion_manager.cuh>
#include <nvbench/cuda_call.cuh>
#include <nvbench/exec_tag.cuh>
#include <nvbench/launch.cuh>
#include <nvbench/state.cuh>
#include <nvbench/stopping_criterion.cuh>
#include <nvbench/type_list.cuh>
#include <nvbench/types.cuh>

#include <cuda_runtime.h>

#include <fmt/format.h>

#include <stdexcept>
#include <string>

#include "test_asserts.cuh"

// Verifies that an exception thrown by a stopping criterion aborts the
// benchmark (the state is marked as failed) instead of being silently swallowed.

namespace
{

__global__ void spin_kernel(nvbench::uint64_t target_cycles)
{
  const auto start = static_cast<nvbench::uint64_t>(clock64());
  while (static_cast<nvbench::uint64_t>(clock64()) - start < target_cycles)
  {
  }
}

constexpr nvbench::uint64_t spin_cycles = 100000;

// Where the criterion should throw from:
enum class throw_site
{
  frequency,
  measurement,
};

// Shared probe so the test can observe how many times the criterion was
// consulted. If the exception were swallowed and sampling continued, these
// counts would climb well past the single call that throws.
struct criterion_probe
{
  throw_site site{throw_site::measurement};
  int frequency_calls{0};
  int measurement_calls{0};
};

criterion_probe g_probe;

// A stopping criterion that throws on demand. The throw happens on the first
// sample, which should abort the run before `do_is_finished()` is ever
// consulted. If `do_is_finished()` *is* reached, the exception must have been
// swallowed, so it returns true to end the run immediately -- this keeps the
// regression case from spinning until the benchmark timeout.
class throwing_criterion final : public nvbench::stopping_criterion_base
{
public:
  throwing_criterion()
      : nvbench::stopping_criterion_base{"test_throwing", {}}
  {}

protected:
  void do_initialize() override {}

  void do_add_frequency(nvbench::float32_t /* frequency_hz */) override
  {
    ++g_probe.frequency_calls;
    if (g_probe.site == throw_site::frequency)
    {
      throw std::runtime_error{"criterion failure from add_frequency"};
    }
  }

  void do_add_measurement(nvbench::float64_t /* measurement */) override
  {
    ++g_probe.measurement_calls;
    if (g_probe.site == throw_site::measurement)
    {
      throw std::runtime_error{"criterion failure from add_measurement"};
    }
  }

  bool do_is_finished() override
  {
    // Only reachable if a sample completed without the throw aborting the run,
    // i.e. the exception was swallowed. Finish immediately so the test fails
    // fast on the is_skipped() check instead of sampling until the timeout.
    return true;
  }
};
NVBENCH_REGISTER_CRITERION(throwing_criterion);

struct spin_generator
{
  void operator()(nvbench::state &state, nvbench::type_list<>) const
  {
    state.exec(nvbench::exec_tag::impl::cold, [](nvbench::launch &launch) {
      spin_kernel<<<1, 1, 0, launch.get_stream()>>>(spin_cycles);
    });
  }
};

using benchmark_type = nvbench::benchmark<spin_generator>;

// Runs a benchmark whose criterion throws from `site`, and asserts that the
// benchmark failed (state skipped with the criterion's error) rather than
// completing.
void run_and_expect_failure(throw_site site)
{
  g_probe      = criterion_probe{};
  g_probe.site = site;

  benchmark_type bench{spin_generator{}};
  bench.add_device(0);
  bench.set_stopping_criterion("test_throwing");

  // Disable throttle detection. Otherwise the unreliable clock reading of this
  // tiny kernel can look like throttling, causing record_measurements() to
  // discard the sample before the criterion is ever consulted -- the throw
  // would never fire and the run would simply time out.
  bench.set_throttle_threshold(0.f);

  bench.run();

  NVBENCH_CUDA_CALL(cudaDeviceSynchronize());

  const auto &states = bench.get_states();
  ASSERT(!states.empty());
  for (const auto &state : states)
  {
    ASSERT(state.is_skipped());
    ASSERT(state.get_skip_reason().find("criterion failure") != std::string::npos);
  }
}

// A throw from `add_measurement` must stop the run after the first sample.
void test_add_measurement_exception_stops_benchmark()
{
  run_and_expect_failure(throw_site::measurement);

  // The skip check above is what proves the run aborted; this confirms the
  // throw happened on the very first measurement,
  // and that a frequency measurement was collected as well.
  ASSERT(g_probe.frequency_calls == 1);
  ASSERT(g_probe.measurement_calls == 1);
}

// A throw from `add_frequency` must stop the run before the measurement for that
// sample is ever recorded.
void test_add_frequency_exception_stops_benchmark()
{
  run_and_expect_failure(throw_site::frequency);

  ASSERT(g_probe.frequency_calls == 1);
  ASSERT(g_probe.measurement_calls == 0);
}

} // namespace

int main()
try
{
  test_add_measurement_exception_stops_benchmark();
  test_add_frequency_exception_stops_benchmark();

  return 0;
}
catch (std::exception &e)
{
  fmt::print("{}\n", e.what());
  return 1;
}
