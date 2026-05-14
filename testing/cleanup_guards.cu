// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <nvbench/blocking_kernel.cuh>
#include <nvbench/cpu_timer.cuh>
#include <nvbench/detail/measure_cold.cuh>
#include <nvbench/detail/measure_cold_launch_timer_core.cuh>
#include <nvbench/detail/measure_hot.cuh>
#include <nvbench/detail/stream_cleanup_guard.cuh>

#include <cuda_runtime_api.h>

#include <fmt/format.h>

#include <array>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "test_asserts.cuh"

namespace
{

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
#if defined(CUDART_VERSION) && CUDART_VERSION > 12600
  // CUDA 12.0 through 12.6 can exhaust host memory in cudafe++ while checking
  // this contract.
  verify_cold_launch_timer_noexcept_contract<cold_launch_timer_probe>();
#endif
  verify_stream_cleanup_guard_noexcept_contract<
    nvbench::detail::stream_cleanup_guard<hot_cleanup_probe>>();

  static_assert(std::is_nothrow_destructible_v<cold_launch_timer_core_probe>,
                "measure_cold_launch_timer_core destructor must remain noexcept.");
}

static_assert((verify_noexcept_contracts(), true), "Noexcept cleanup contracts must hold.");

enum class action
{
  flush_device_l2,
  sync_stream,
  sync_stream_noexcept,
  cpu_timer_start,
  cpu_timer_stop,
  cpu_timer_stop_noexcept,
  block_stream,
  unblock_stream,
  unblock_stream_noexcept,
  gpu_frequency_start,
  gpu_frequency_stop,
  profiler_start,
  profiler_stop,
  profiler_stop_noexcept,
  cuda_timer_start,
  cuda_timer_stop,
};

struct fake_measure
{
  void clear_actions() noexcept
  {
    action_count = 0;
    overflow     = false;
  }

  void throw_on(action a) noexcept
  {
    should_throw = true;
    throw_action = a;
  }

  void record(action a) noexcept
  {
    if (action_count < actions.size())
    {
      actions[action_count++] = a;
    }
    else
    {
      overflow = true;
    }
  }

  void record_or_throw(action a)
  {
    this->record(a);
    if (should_throw && throw_action == a)
    {
      should_throw = false;
      throw std::runtime_error{"Injected fake_measure failure."};
    }
  }

  void flush_device_l2() { this->record_or_throw(action::flush_device_l2); }
  void sync_stream() { this->record_or_throw(action::sync_stream); }
  int sync_stream_noexcept() noexcept
  {
    this->record(action::sync_stream_noexcept);
    return 0;
  }

  void cpu_timer_start() noexcept { this->record(action::cpu_timer_start); }
  void cpu_timer_stop() noexcept { this->record(action::cpu_timer_stop); }
  void cpu_timer_stop_noexcept() noexcept { this->record(action::cpu_timer_stop_noexcept); }

  void block_stream() { this->record_or_throw(action::block_stream); }
  void unblock_stream() { this->record_or_throw(action::unblock_stream); }
  void unblock_stream_noexcept() noexcept { this->record(action::unblock_stream_noexcept); }

  void gpu_frequency_start() { this->record_or_throw(action::gpu_frequency_start); }
  void gpu_frequency_stop() { this->record_or_throw(action::gpu_frequency_stop); }

  void profiler_start() { this->record_or_throw(action::profiler_start); }
  void profiler_stop() { this->record_or_throw(action::profiler_stop); }
  int profiler_stop_noexcept() noexcept
  {
    this->record(action::profiler_stop_noexcept);
    return 0;
  }

  void cuda_timer_start() { this->record_or_throw(action::cuda_timer_start); }
  void cuda_timer_stop() { this->record_or_throw(action::cuda_timer_stop); }

  std::array<action, 32> actions{};
  std::size_t action_count{};
  action throw_action{};
  bool should_throw{false};
  bool overflow{false};
};

template <typename Callable>
void assert_throws(Callable &&callable)
{
  bool threw = false;
  try
  {
    callable();
  }
  catch (const std::runtime_error &)
  {
    threw = true;
  }
  ASSERT(threw);
}

void assert_actions(const fake_measure &measure, std::initializer_list<action> expected)
{
  ASSERT(!measure.overflow);
  ASSERT(measure.action_count == expected.size());

  std::size_t index = 0;
  for (const action expected_action : expected)
  {
    ASSERT(measure.actions[index] == expected_action);
    ++index;
  }
}

void test_stream_cleanup_guard_block_stream_throw()
{
  fake_measure measure;
  measure.throw_on(action::block_stream);

  assert_throws([&measure] {
    nvbench::detail::stream_cleanup_guard<fake_measure> cleanup{measure};
    cleanup.block_stream();
  });

  assert_actions(
    measure,
    {action::block_stream, action::unblock_stream_noexcept, action::sync_stream_noexcept});
}

void test_stream_cleanup_guard_unblock_then_throw()
{
  fake_measure measure;

  assert_throws([&measure] {
    nvbench::detail::stream_cleanup_guard<fake_measure> cleanup{measure};
    cleanup.block_stream();
    cleanup.unblock();
    throw std::runtime_error{"Injected post-unblock failure."};
  });

  assert_actions(measure,
                 {action::block_stream, action::unblock_stream, action::sync_stream_noexcept});
}

void test_kernel_launch_timer_block_stream_throw()
{
  fake_measure measure;
  measure.throw_on(action::block_stream);

  assert_throws([&measure] {
    nvbench::detail::measure_cold_launch_timer_core<fake_measure> timer{
      measure,
      nvbench::detail::measure_cold_launch_timer_config{false, false, true}};
    timer.start();
  });

  assert_actions(measure,
                 {action::flush_device_l2,
                  action::sync_stream,
                  action::cpu_timer_start,
                  action::block_stream,
                  action::unblock_stream_noexcept,
                  action::sync_stream_noexcept,
                  action::cpu_timer_stop_noexcept});
}

void test_kernel_launch_timer_gpu_frequency_start_throw()
{
  fake_measure measure;
  measure.throw_on(action::gpu_frequency_start);

  assert_throws([&measure] {
    nvbench::detail::measure_cold_launch_timer_core<fake_measure> timer{
      measure,
      nvbench::detail::measure_cold_launch_timer_config{false, false, true}};
    timer.start();
  });

  assert_actions(measure,
                 {action::flush_device_l2,
                  action::sync_stream,
                  action::cpu_timer_start,
                  action::block_stream,
                  action::gpu_frequency_start,
                  action::unblock_stream_noexcept,
                  action::sync_stream_noexcept,
                  action::cpu_timer_stop_noexcept});
}

void test_kernel_launch_timer_gpu_frequency_stop_throw()
{
  fake_measure measure;
  nvbench::detail::measure_cold_launch_timer_core<fake_measure> timer{
    measure,
    nvbench::detail::measure_cold_launch_timer_config{false, false, true}};

  timer.start();
  measure.clear_actions();
  measure.throw_on(action::gpu_frequency_stop);

  assert_throws([&timer] { timer.stop(); });

  assert_actions(measure,
                 {action::cuda_timer_stop,
                  action::gpu_frequency_stop,
                  action::unblock_stream_noexcept,
                  action::sync_stream_noexcept,
                  action::cpu_timer_stop_noexcept});
}

} // namespace

int main()
try
{
  test_stream_cleanup_guard_block_stream_throw();
  test_stream_cleanup_guard_unblock_then_throw();
  test_kernel_launch_timer_block_stream_throw();
  test_kernel_launch_timer_gpu_frequency_start_throw();
  test_kernel_launch_timer_gpu_frequency_stop_throw();

  return 0;
}
catch (std::exception &e)
{
  fmt::print("{}\n", e.what());
  return 1;
}
