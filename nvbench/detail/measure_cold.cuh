/*
 *  Copyright 2021 NVIDIA Corporation
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

#include <nvbench/blocking_kernel.cuh>
#include <nvbench/cpu_timer.cuh>
#include <nvbench/cuda_call.cuh>
#include <nvbench/cuda_timer.cuh>
#include <nvbench/detail/gpu_frequency.cuh>
#include <nvbench/detail/kernel_launcher_timer_wrapper.cuh>
#include <nvbench/detail/l2flush.cuh>
#include <nvbench/detail/statistics.cuh>
#include <nvbench/device_info.cuh>
#include <nvbench/exec_tag.cuh>
#include <nvbench/launch.cuh>
#include <nvbench/stopping_criterion.cuh>
#include <nvbench/types.cuh>

#include <cuda_runtime.h>

#include <utility>
#include <vector>

namespace nvbench
{

struct state;

namespace detail
{

// non-templated code goes here:
struct measure_cold_base
{
  explicit measure_cold_base(nvbench::state &exec_state);
  measure_cold_base(const measure_cold_base &)            = delete;
  measure_cold_base(measure_cold_base &&)                 = delete;
  measure_cold_base &operator=(const measure_cold_base &) = delete;
  measure_cold_base &operator=(measure_cold_base &&)      = delete;

protected:
  struct kernel_launch_timer;
  friend struct kernel_launch_timer;

  void check();
  void initialize();
  void run_trials_prologue();
  void record_measurements();
  bool is_finished();
  void run_trials_epilogue();
  void generate_summaries();
  void gpu_frequency_start() { m_gpu_frequency.start(m_launch.get_stream()); }
  void gpu_frequency_stop() { m_gpu_frequency.stop(m_launch.get_stream()); }

  void check_skip_time(nvbench::float64_t warmup_time);

  __forceinline__ void flush_device_l2() { m_l2flush.flush(m_launch.get_stream()); }

  __forceinline__ void sync_stream() const
  {
    NVBENCH_CUDA_CALL(cudaStreamSynchronize(m_launch.get_stream()));
  }

  void block_stream();
  __forceinline__ void unblock_stream() { m_blocker.unblock(); }

  nvbench::state &m_state;

  nvbench::launch m_launch;
  nvbench::cuda_timer m_cuda_timer;
  nvbench::cpu_timer m_cpu_timer;
  nvbench::cpu_timer m_walltime_timer;
  nvbench::detail::l2flush m_l2flush;
  nvbench::blocking_kernel m_blocker;

  nvbench::criterion_params m_criterion_params;
  nvbench::stopping_criterion_base &m_stopping_criterion;
  nvbench::detail::gpu_frequency m_gpu_frequency;

  bool m_disable_blocking_kernel{false};
  bool m_run_once{false};
  bool m_check_throttling;

  nvbench::int64_t m_min_samples{};

  nvbench::float64_t m_skip_time{};
  nvbench::float64_t m_timeout{};

  nvbench::float32_t m_throttle_threshold;      // [% of default SM clock rate]
  nvbench::float32_t m_throttle_recovery_delay; // [seconds]

  // Dynamically increased when repeated throttling occurs
  // without successfully recording a sample.
  nvbench::float32_t m_dynamic_throttle_recovery_delay{}; // [seconds]
  nvbench::int64_t m_throttle_discard_count{};

  nvbench::int64_t m_total_samples{};

  nvbench::float64_t m_min_cuda_time{};
  nvbench::float64_t m_max_cuda_time{};
  nvbench::float64_t m_total_cuda_time{};

  nvbench::float64_t m_min_cpu_time{};
  nvbench::float64_t m_max_cpu_time{};
  nvbench::float64_t m_total_cpu_time{};

  nvbench::float64_t m_sm_clock_rate_accumulator{};

  std::vector<nvbench::float64_t> m_cuda_times;
  std::vector<nvbench::float64_t> m_cpu_times;

  bool m_max_time_exceeded{};
};

struct measure_cold_base::kernel_launch_timer
{
  kernel_launch_timer(measure_cold_base &measure)
      : m_measure{measure}
      , m_disable_blocking_kernel{measure.m_disable_blocking_kernel}
  {}

  explicit kernel_launch_timer(measure_cold_base &measure, bool disable_blocking_kernel)
      : m_measure{measure}
      , m_disable_blocking_kernel{disable_blocking_kernel}
  {}

  __forceinline__ void start()
  {
    m_measure.flush_device_l2();
    m_measure.sync_stream();
    if (!m_disable_blocking_kernel)
    {
      m_measure.block_stream();
    }
    if (m_measure.m_check_throttling)
    {
      m_measure.gpu_frequency_start();
    }
    m_measure.m_cuda_timer.start(m_measure.m_launch.get_stream());
    // start CPU timer irrespective of use of blocking kernel
    // Ref: https://github.com/NVIDIA/nvbench/issues/249
    m_measure.m_cpu_timer.start();
  }

  __forceinline__ void stop()
  {
    m_measure.m_cuda_timer.stop(m_measure.m_launch.get_stream());
    if (!m_disable_blocking_kernel)
    {
      m_measure.unblock_stream();
    }
    if (m_measure.m_check_throttling)
    {
      m_measure.gpu_frequency_stop();
    }
    m_measure.sync_stream();
    m_measure.m_cpu_timer.stop();
  }

private:
  measure_cold_base &m_measure;
  bool m_disable_blocking_kernel;
};

template <typename KernelLauncher>
struct measure_cold : public measure_cold_base
{
  measure_cold(nvbench::state &state, KernelLauncher &kernel_launcher)
      : measure_cold_base(state)
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
  // Run the kernel once, measuring the GPU time. If under skip_time, skip the
  // measurement.
  void run_warmup()
  {
    if (m_run_once)
    { // Skip warmups
      return;
    }

    // disable use of blocking kernel for warm-up run
    // see https://github.com/NVIDIA/nvbench/issues/240
    constexpr bool disable_blocking_kernel = true;
    kernel_launch_timer timer(*this, disable_blocking_kernel);

    this->launch_kernel(timer);
    this->check_skip_time(m_cuda_timer.get_duration());
  }

  void run_trials()
  {
    // do not use blocking kernel if benchmark is only run once, e.g., when profiling
    // ref: https://github.com/NVIDIA/nvbench/issue/242
    const bool disable_blocking_kernel = m_run_once || m_disable_blocking_kernel;
    kernel_launch_timer timer(*this, disable_blocking_kernel);
    do
    {
      this->launch_kernel(timer);
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
