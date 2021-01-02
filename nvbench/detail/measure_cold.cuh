#pragma once

#include <nvbench/cpu_timer.cuh>
#include <nvbench/cuda_call.cuh>
#include <nvbench/cuda_timer.cuh>
#include <nvbench/launch.cuh>
#include <nvbench/state.cuh>

#include <nvbench/detail/l2flush.cuh>

#include <cuda_runtime.h>

#include <utility>

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
  void initialize();

  void generate_summaries();

  nvbench::launch m_launch{};
  nvbench::cuda_timer m_cuda_timer{};
  nvbench::cpu_timer m_cpu_timer{};
  nvbench::detail::l2flush m_l2flush{};

  // seconds:
  nvbench::float64_t m_min_time{1.};
  nvbench::float64_t m_cuda_time{};
  nvbench::float64_t m_cpu_time{};

  nvbench::int64_t m_num_trials{};

  nvbench::state &m_state;
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
    do
    {
      m_l2flush.flush(m_launch.get_stream());
      NVBENCH_CUDA_CALL(cudaStreamSynchronize(m_launch.get_stream()));

      m_cuda_timer.start(m_launch.get_stream());
      m_cpu_timer.start();

      this->launch_kernel();

      m_cuda_timer.stop(m_launch.get_stream());

      NVBENCH_CUDA_CALL(cudaStreamSynchronize(m_launch.get_stream()));
      m_cpu_timer.stop();

      // TODO eventually these should also get logged in a vector for
      // statistical analysis.
      m_cuda_time += m_cuda_timer.get_duration();
      m_cpu_time += m_cpu_timer.get_duration();
      ++m_num_trials;
    } while (std::max(m_cuda_time, m_cpu_time) < m_min_time);
  }

  // TODO forceinline
  void launch_kernel() { m_kernel_launcher(m_launch); }

  KernelLauncher &m_kernel_launcher;
};

} // namespace detail
} // namespace nvbench
