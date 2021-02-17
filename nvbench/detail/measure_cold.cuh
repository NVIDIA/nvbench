#pragma once

#include <nvbench/blocking_kernel.cuh>
#include <nvbench/cpu_timer.cuh>
#include <nvbench/cuda_call.cuh>
#include <nvbench/cuda_timer.cuh>
#include <nvbench/device_info.cuh>
#include <nvbench/exec_tag.cuh>
#include <nvbench/launch.cuh>

#include <nvbench/detail/l2flush.cuh>
#include <nvbench/detail/statistics.cuh>

#include <cuda_runtime.h>

#include <algorithm>
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
  measure_cold_base(const measure_cold_base &) = delete;
  measure_cold_base(measure_cold_base &&)      = delete;
  measure_cold_base &operator=(const measure_cold_base &) = delete;
  measure_cold_base &operator=(measure_cold_base &&) = delete;

protected:
  void check();

  void initialize()
  {
    m_total_cuda_time = 0.;
    m_total_cpu_time  = 0.;
    m_cuda_noise      = 0.;
    m_cpu_noise       = 0.;
    m_total_samples   = 0;
    m_cuda_times.clear();
    m_cpu_times.clear();
    m_max_time_exceeded = false;
  }

  void generate_summaries();

  void check_skip_time(nvbench::float64_t warmup_time);

  nvbench::state &m_state;

  nvbench::launch m_launch;
  nvbench::cuda_timer m_cuda_timer;
  nvbench::cpu_timer m_cpu_timer;
  nvbench::cpu_timer m_timeout_timer;
  nvbench::detail::l2flush m_l2flush;

  nvbench::int64_t m_min_samples{};
  nvbench::float64_t m_max_noise{}; // % rel stdev
  nvbench::float64_t m_min_time{};

  nvbench::float64_t m_skip_time{};
  nvbench::float64_t m_timeout{};

  nvbench::int64_t m_total_samples{};
  nvbench::float64_t m_total_cuda_time{};
  nvbench::float64_t m_total_cpu_time{};
  nvbench::float64_t m_cuda_noise{}; // % rel stdev
  nvbench::float64_t m_cpu_noise{};  // % rel stdev

  std::vector<nvbench::float64_t> m_cuda_times;
  std::vector<nvbench::float64_t> m_cpu_times;

  bool m_max_time_exceeded{};
};

template <typename KernelLauncher, nvbench::detail::exec_flag ExecTagModifiers>
struct measure_cold : public measure_cold_base
{
  static constexpr bool needs_timer_wrapper =
    (ExecTagModifiers & nvbench::detail::exec_flag::timer) ==
    nvbench::detail::exec_flag::none;
  static constexpr bool use_blocking_kernel =
    (ExecTagModifiers & nvbench::detail::exec_flag::no_block) ==
    nvbench::detail::exec_flag::none;

  measure_cold(nvbench::state &state, KernelLauncher &kernel_launcher)
      : measure_cold_base(state)
      , m_kernel_launcher{kernel_launcher}
  {}

  void operator()()
  {
    this->check();
    this->initialize();
    this->run_warmup();
    this->run_trials();
    this->generate_summaries();
  }

private:
  // Run the kernel once, measuring the GPU time. If under skip_time, skip the
  // measurement.
  void run_warmup()
  {
    this->flush_device_l2();
    this->sync_stream();

    nvbench::blocking_kernel blocker;
    if constexpr (use_blocking_kernel)
    {
      blocker.block(m_launch.get_stream());
    }

    m_cuda_timer.start(m_launch.get_stream());
    this->launch_kernel();
    m_cuda_timer.stop(m_launch.get_stream());

    if constexpr (use_blocking_kernel)
    {
      blocker.unblock();
    }
    this->sync_stream();

    this->check_skip_time(m_cuda_timer.get_duration());
  }

  void run_trials()
  {
    m_timeout_timer.start();
    nvbench::blocking_kernel blocker;
    do
    {
      this->flush_device_l2();
      this->sync_stream();

      if constexpr (use_blocking_kernel)
      {
        blocker.block(m_launch.get_stream());
      }
      else
      {
        m_cpu_timer.start();
      }

      m_cuda_timer.start(m_launch.get_stream());
      this->launch_kernel();
      m_cuda_timer.stop(m_launch.get_stream());

      if constexpr (use_blocking_kernel)
      {
        m_cpu_timer.start();
        blocker.unblock();
      }
      this->sync_stream();
      m_cpu_timer.stop();

      const auto cur_cuda_time = m_cuda_timer.get_duration();
      const auto cur_cpu_time  = m_cpu_timer.get_duration();
      m_cuda_times.push_back(cur_cuda_time);
      m_cpu_times.push_back(cur_cpu_time);
      m_total_cuda_time += cur_cuda_time;
      m_total_cpu_time += cur_cpu_time;
      ++m_total_samples;

      // Only consider the cuda noise in the convergence criteria.
      m_cuda_noise = nvbench::detail::compute_noise(m_cuda_times,
                                                    m_total_cuda_time);

      m_timeout_timer.stop();
      const auto total_time = m_timeout_timer.get_duration();

      if (m_total_cuda_time > m_min_time &&  // Min time okay
          m_total_samples > m_min_samples && // Min samples okay
          m_cuda_noise < m_max_noise)        // Noise okay
      {
        break;
      }

      if (total_time > m_timeout) // Max time exceeded, stop iterating.
      {
        m_max_time_exceeded = true;
        break;
      }
    } while (true);
    m_cpu_noise = nvbench::detail::compute_noise(m_cpu_times, m_total_cpu_time);
  }

  __forceinline__ void flush_device_l2()
  {
    m_l2flush.flush(m_launch.get_stream());
  }

  __forceinline__ void launch_kernel() { m_kernel_launcher(m_launch); }

  __forceinline__ void sync_stream() const
  {
    NVBENCH_CUDA_CALL(cudaStreamSynchronize(m_launch.get_stream()));
  }

  KernelLauncher &m_kernel_launcher;
};

} // namespace detail
} // namespace nvbench
