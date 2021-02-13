#pragma once

#include <nvbench/blocking_kernel.cuh>
#include <nvbench/cpu_timer.cuh>
#include <nvbench/cuda_call.cuh>
#include <nvbench/cuda_timer.cuh>
#include <nvbench/device_info.cuh>
#include <nvbench/launch.cuh>
#include <nvbench/state.cuh>

#include <nvbench/detail/l2flush.cuh>
#include <nvbench/detail/statistics.cuh>

#include <cuda_runtime.h>

#include <algorithm>
#include <utility>
#include <vector>

namespace nvbench
{

namespace detail
{

// non-templated code goes here:
struct measure_cold_base
{
  explicit measure_cold_base(nvbench::state &exec_state)
      : m_state(exec_state)
  {}
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
    m_total_iters     = 0;
    m_cuda_times.clear();
    m_cpu_times.clear();
    m_max_time_exceeded = false;
  }

  void generate_summaries();

  nvbench::state &m_state;

  nvbench::launch m_launch;
  nvbench::cuda_timer m_cuda_timer;
  nvbench::cpu_timer m_cpu_timer;
  nvbench::cpu_timer m_timeout_timer;
  nvbench::detail::l2flush m_l2flush;

  nvbench::int64_t m_min_iters{10};
  nvbench::int64_t m_total_iters{};

  nvbench::float64_t m_max_noise{0.5}; // % rel stdev
  nvbench::float64_t m_cuda_noise{};   // % rel stdev
  nvbench::float64_t m_cpu_noise{};    // % rel stdev

  nvbench::float64_t m_min_time{0.5};
  nvbench::float64_t m_max_time{5.0};

  nvbench::float64_t m_total_cuda_time{};
  nvbench::float64_t m_total_cpu_time{};

  std::vector<nvbench::float64_t> m_cuda_times;
  std::vector<nvbench::float64_t> m_cpu_times;

  bool m_max_time_exceeded{};
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
    this->run_trials();
    this->generate_summaries();
  }

private:
  void run_warmup()
  {
    m_l2flush.flush(m_launch.get_stream());
    this->launch_kernel();
    NVBENCH_CUDA_CALL(cudaStreamSynchronize(m_launch.get_stream()));
  }

  void run_trials()
  {
    m_timeout_timer.start();
    nvbench::blocking_kernel blocker;
    do
    {
      m_l2flush.flush(m_launch.get_stream());
      NVBENCH_CUDA_CALL(cudaStreamSynchronize(m_launch.get_stream()));

      blocker.block(m_launch.get_stream());
      m_cuda_timer.start(m_launch.get_stream());
      this->launch_kernel();
      m_cuda_timer.stop(m_launch.get_stream());

      m_cpu_timer.start();
      blocker.release();
      NVBENCH_CUDA_CALL(cudaStreamSynchronize(m_launch.get_stream()));
      m_cpu_timer.stop();

      const auto cur_cuda_time = m_cuda_timer.get_duration();
      const auto cur_cpu_time  = m_cpu_timer.get_duration();
      m_cuda_times.push_back(cur_cuda_time);
      m_cpu_times.push_back(cur_cpu_time);
      m_total_cuda_time += cur_cuda_time;
      m_total_cpu_time += cur_cpu_time;
      ++m_total_iters;

      // Only consider the cuda noise in the convergence criteria.
      m_cuda_noise = nvbench::detail::compute_noise(m_cuda_times,
                                                    m_total_cuda_time);

      m_timeout_timer.stop();
      const auto total_time = m_timeout_timer.get_duration();

      if (m_total_cuda_time > m_min_time && // Min time okay
          m_total_iters > m_min_iters &&    // Min iters okay
          m_cuda_noise < m_max_noise)       // Noise okay
      {
        break;
      }

      if (total_time > m_max_time) // Max time exceeded, stop iterating.
      {
        m_max_time_exceeded = true;
        break;
      }
    } while (true);
    m_cpu_noise = nvbench::detail::compute_noise(m_cpu_times, m_total_cpu_time);
  }

  __forceinline__ void launch_kernel() { m_kernel_launcher(m_launch); }

  KernelLauncher &m_kernel_launcher;
};

} // namespace detail
} // namespace nvbench
