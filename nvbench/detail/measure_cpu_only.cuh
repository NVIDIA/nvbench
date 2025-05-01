/*
 *  Copyright 2021-2025 NVIDIA Corporation
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

#pragma once

#include <nvbench/cpu_timer.cuh>
#include <nvbench/detail/kernel_launcher_timer_wrapper.cuh>
#include <nvbench/detail/statistics.cuh>
#include <nvbench/exec_tag.cuh>
#include <nvbench/launch.cuh>
#include <nvbench/stopping_criterion.cuh>

#include <utility>
#include <vector>

namespace nvbench
{

struct state;

namespace detail
{

// non-templated code goes here:
struct measure_cpu_only_base
{
  explicit measure_cpu_only_base(nvbench::state &exec_state);
  measure_cpu_only_base(const measure_cpu_only_base &)            = delete;
  measure_cpu_only_base(measure_cpu_only_base &&)                 = delete;
  measure_cpu_only_base &operator=(const measure_cpu_only_base &) = delete;
  measure_cpu_only_base &operator=(measure_cpu_only_base &&)      = delete;

protected:
  void check();
  void initialize();
  void run_trials_prologue();
  void record_measurements();
  bool is_finished();
  void run_trials_epilogue();
  void generate_summaries();

  void check_skip_time(nvbench::float64_t warmup_time);

  nvbench::state &m_state;

  // Required to satisfy the KernelLauncher interface:
  nvbench::launch m_launch;

  nvbench::cpu_timer m_cpu_timer;
  nvbench::cpu_timer m_walltime_timer;

  nvbench::criterion_params m_criterion_params;
  nvbench::stopping_criterion_base &m_stopping_criterion;

  bool m_run_once{false};

  nvbench::int64_t m_min_samples{};

  nvbench::float64_t m_skip_time{};
  nvbench::float64_t m_timeout{};

  nvbench::int64_t m_total_samples{};

  nvbench::float64_t m_min_cpu_time{};
  nvbench::float64_t m_max_cpu_time{};
  nvbench::float64_t m_total_cpu_time{};

  std::vector<nvbench::float64_t> m_cpu_times;

  bool m_max_time_exceeded{};
};

template <typename KernelLauncher>
struct measure_cpu_only : public measure_cpu_only_base
{
  measure_cpu_only(nvbench::state &state, KernelLauncher &kernel_launcher)
      : measure_cpu_only_base(state)
      , m_kernel_launcher{kernel_launcher}
  {}

  void operator()()
  {
    this->check();
    this->initialize();
    this->run_warmup();

    this->run_trials_prologue();
    this->run_trials();
    this->run_trials_epilogue();

    this->generate_summaries();
  }

private:
  // Run the kernel once, measuring the CPU time. If under skip_time, skip the
  // measurement.
  void run_warmup()
  {
    if (m_run_once)
    { // Skip warmups
      return;
    }

    this->launch_kernel(m_cpu_timer);
    this->check_skip_time(m_cpu_timer.get_duration());
  }

  void run_trials()
  {
    do
    {
      this->launch_kernel(m_cpu_timer);
      this->record_measurements();
    } while (!this->is_finished());
  }

  template <typename TimerT>
  __forceinline__ void launch_kernel(TimerT &timer)
  {
    m_kernel_launcher(m_launch, timer);
  }

  KernelLauncher &m_kernel_launcher;
};

} // namespace detail
} // namespace nvbench
