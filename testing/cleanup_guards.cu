// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <nvbench/detail/measure_cold_launch_timer_core.cuh>
#include <nvbench/detail/stream_cleanup_guard.cuh>

#include <fmt/format.h>

#include <array>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>

#include "test_asserts.cuh"

namespace
{

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
