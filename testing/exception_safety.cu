// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <nvbench/benchmark.cuh>
#include <nvbench/cuda_call.cuh>
#include <nvbench/detail/measure_cold.cuh>
#include <nvbench/detail/measure_cold_launch_timer_core.cuh>
#include <nvbench/detail/measure_hot.cuh>
#include <nvbench/detail/stream_cleanup_guard.cuh>
#include <nvbench/exec_tag.cuh>
#include <nvbench/launch.cuh>
#include <nvbench/runner.cuh>
#include <nvbench/state.cuh>
#include <nvbench/type_list.cuh>
#include <nvbench/types.cuh>

#include <cuda_runtime.h>

#include <fmt/format.h>

#include <chrono>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "test_asserts.cuh"

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

struct hot_cleanup_probe : nvbench::detail::measure_hot_base
{
  using nvbench::detail::measure_hot_base::sync_stream_noexcept;
  using nvbench::detail::measure_hot_base::unblock_stream_noexcept;
};

struct cold_cleanup_probe : nvbench::detail::measure_cold_base
{
  using kernel_launch_timer = nvbench::detail::measure_cold_base::kernel_launch_timer;

  using nvbench::detail::measure_cold_base::profiler_stop_noexcept;
  using nvbench::detail::measure_cold_base::sync_stream_noexcept;
  using nvbench::detail::measure_cold_base::unblock_stream_noexcept;
};

using cold_launch_timer_probe = cold_cleanup_probe::kernel_launch_timer;
using cold_launch_timer_core_probe =
  nvbench::detail::measure_cold_launch_timer_core<cold_launch_timer_probe>;

template <typename Timer>
constexpr void verify_cpu_timer_noexcept_contract()
{
  static_assert(noexcept(std::declval<Timer &>().start()),
                "CPU timer start must remain noexcept for cleanup-safe measurement code.");
  static_assert(noexcept(std::declval<Timer &>().stop()),
                "CPU timer stop must remain noexcept for cleanup-safe measurement code.");
}

template <typename Measure>
constexpr void verify_stream_cleanup_measure_noexcept_contract()
{
  static_assert(noexcept(std::declval<Measure &>().sync_stream_noexcept()),
                "Cleanup measure sync_stream_noexcept must remain noexcept.");
  static_assert(noexcept(std::declval<Measure &>().unblock_stream_noexcept()),
                "Cleanup measure unblock_stream_noexcept must remain noexcept.");
}

template <typename Measure>
constexpr void verify_cold_measure_noexcept_contract()
{
  verify_stream_cleanup_measure_noexcept_contract<Measure>();
  static_assert(noexcept(std::declval<Measure &>().profiler_stop_noexcept()),
                "Cold cleanup measure profiler_stop_noexcept must remain noexcept.");
}

template <typename Timer>
constexpr void verify_cold_launch_timer_noexcept_contract()
{
  static_assert(noexcept(std::declval<Timer &>().cpu_timer_start()),
                "Cold kernel_launch_timer cpu_timer_start must remain noexcept.");
  static_assert(noexcept(std::declval<Timer &>().cpu_timer_stop()),
                "Cold kernel_launch_timer cpu_timer_stop must remain noexcept.");
  static_assert(noexcept(std::declval<Timer &>().cpu_timer_stop_noexcept()),
                "Cold kernel_launch_timer cpu_timer_stop_noexcept must remain noexcept.");
  static_assert(noexcept(std::declval<Timer &>().sync_stream_noexcept()),
                "Cold kernel_launch_timer sync_stream_noexcept must remain noexcept.");
  static_assert(noexcept(std::declval<Timer &>().profiler_stop_noexcept()),
                "Cold kernel_launch_timer profiler_stop_noexcept must remain noexcept.");
  static_assert(noexcept(std::declval<Timer &>().unblock_stream_noexcept()),
                "Cold kernel_launch_timer unblock_stream_noexcept must remain noexcept.");
  static_assert(std::is_nothrow_destructible_v<Timer>,
                "Cold kernel_launch_timer destructor must remain noexcept.");
}

template <typename Guard>
constexpr void verify_stream_cleanup_guard_noexcept_contract()
{
  static_assert(std::is_nothrow_destructible_v<Guard>,
                "stream_cleanup_guard destructor must remain noexcept.");
  static_assert(noexcept(std::declval<Guard &>().release()),
                "stream_cleanup_guard release must remain noexcept.");
}

constexpr void verify_noexcept_contracts()
{
  verify_cpu_timer_noexcept_contract<nvbench::cpu_timer>();
  static_assert(noexcept(std::declval<nvbench::blocking_kernel &>().unblock_noexcept()),
                "blocking_kernel unblock_noexcept must remain noexcept.");

  verify_stream_cleanup_measure_noexcept_contract<hot_cleanup_probe>();
  verify_cold_measure_noexcept_contract<cold_cleanup_probe>();
  verify_cold_launch_timer_noexcept_contract<cold_launch_timer_probe>();
  verify_stream_cleanup_guard_noexcept_contract<
    nvbench::detail::stream_cleanup_guard<hot_cleanup_probe>>();

  static_assert(std::is_nothrow_destructible_v<cold_launch_timer_core_probe>,
                "measure_cold_launch_timer_core destructor must remain noexcept.");
}

static_assert((verify_noexcept_contracts(), true), "Noexcept cleanup contracts must hold.");

enum class measurement_kind
{
  cold,
  hot,
};

enum class exception_kind
{
  runtime_error,
  stop_runner_loop,
};

struct test_control
{
  measurement_kind measurement{measurement_kind::cold};
  exception_kind exception{exception_kind::runtime_error};
  int generator_calls{0};
  int launcher_calls{0};
};

void synchronize_with_timeout_guard()
{
  const auto start = std::chrono::steady_clock::now();
  NVBENCH_CUDA_CALL(cudaDeviceSynchronize());
  const auto elapsed = std::chrono::steady_clock::now() - start;

  ASSERT_MSG(elapsed < std::chrono::seconds{5},
             "cudaDeviceSynchronize took {} ms; stream cleanup may have leaked blocked work",
             std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count());
}

void throw_requested_exception(exception_kind exception)
{
  if (exception == exception_kind::stop_runner_loop)
  {
    throw nvbench::stop_runner_loop{"requested stop from exception-safety test"};
  }

  throw std::runtime_error{"requested throw from exception-safety test"};
}

void configure_state(nvbench::state &state)
{
  state.set_min_samples(1);
  state.set_timeout(0.01);

  // Keep this below the CTest timeout. If cleanup fails to unblock the
  // blocking kernel, the device-side timeout lets the elapsed-time assertion
  // report the leak before CTest has to kill the process.
  state.set_blocking_kernel_timeout(10.0);
}

void run_throwing_measurement(nvbench::state &state, test_control &control)
{
  configure_state(state);

  auto launcher = [&control](nvbench::launch &launch) {
    ++control.launcher_calls;
    spin_kernel<<<1, 1, 0, launch.get_stream()>>>(spin_cycles);

    // Let the warmup complete. The next launcher call happens under the cold
    // or hot measurement cleanup path that this test is exercising.
    if (control.launcher_calls > 1)
    {
      throw_requested_exception(control.exception);
    }
  };

  if (control.measurement == measurement_kind::hot)
  {
    state.exec(nvbench::exec_tag::impl::hot, launcher);
  }
  else
  {
    state.exec(nvbench::exec_tag::impl::cold, launcher);
  }
}

struct throwing_generator
{
  test_control *control{};

  void operator()(nvbench::state &state, nvbench::type_list<>) const
  {
    ++control->generator_calls;
    run_throwing_measurement(state, *control);
  }
};

using benchmark_type = nvbench::benchmark<throwing_generator>;

void run_benchmark(test_control &control, bool add_axis = false)
{
  benchmark_type bench{throwing_generator{&control}};
  bench.add_device(0);
  bench.set_min_samples(1);
  bench.set_timeout(0.01);
  bench.set_criterion_param_float64("min-time", 1e-6);
  if (add_axis)
  {
    bench.add_int64_axis("Case", {0, 1, 2});
  }

  bench.run();

  synchronize_with_timeout_guard();

  ASSERT(!bench.get_states().empty());
  ASSERT(bench.get_states().front().is_skipped());
  ASSERT(bench.get_states().front().get_skip_reason().find("requested") != std::string::npos);
}

void test_cold_runtime_error_cleanup()
{
  test_control control;
  control.measurement = measurement_kind::cold;
  control.exception   = exception_kind::runtime_error;

  run_benchmark(control);

  ASSERT(control.generator_calls == 1);
  ASSERT(control.launcher_calls == 2);
}

void test_hot_runtime_error_cleanup()
{
  test_control control;
  control.measurement = measurement_kind::hot;
  control.exception   = exception_kind::runtime_error;

  run_benchmark(control);

  ASSERT(control.generator_calls == 1);
  ASSERT(control.launcher_calls == 2);
}

void test_stop_runner_loop_cleanup_and_skip_remaining()
{
  test_control control;
  control.measurement = measurement_kind::cold;
  control.exception   = exception_kind::stop_runner_loop;

  run_benchmark(control, true);

  ASSERT(control.generator_calls == 1);
  ASSERT(control.launcher_calls == 2);
}

} // namespace

int main()
try
{
  test_cold_runtime_error_cleanup();
  test_hot_runtime_error_cleanup();
  test_stop_runner_loop_cleanup_and_skip_remaining();

  return 0;
}
catch (std::exception &e)
{
  fmt::print("{}\n", e.what());
  return 1;
}
