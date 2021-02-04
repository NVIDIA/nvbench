#pragma once

#include <nvbench/cpu_timer.cuh>
#include <nvbench/cuda_call.cuh>
#include <nvbench/cuda_timer.cuh>
#include <nvbench/launch.cuh>
#include <nvbench/state.cuh>

#include <cuda_runtime.h>

#include <utility>

namespace nvbench
{

namespace detail
{

// non-templated code goes here:
struct measure_hot_base
{
  explicit measure_hot_base(nvbench::state &exec_state)
      : m_state(exec_state)
  {}
  measure_hot_base(const measure_hot_base &) = delete;
  measure_hot_base(measure_hot_base &&)      = delete;
  measure_hot_base &operator=(const measure_hot_base &) = delete;
  measure_hot_base &operator=(measure_hot_base &&) = delete;

protected:
  void initialize()
  {
    m_cuda_time  = 0.;
    m_cpu_time   = 0.;
    m_num_trials = 0;
  }

  void generate_summaries();

  nvbench::launch m_launch{};
  nvbench::cuda_timer m_cuda_timer{};
  nvbench::cpu_timer m_cpu_timer{};

  // seconds:
  nvbench::float64_t m_min_time{1.};
  nvbench::float64_t m_cuda_time{};
  nvbench::float64_t m_cpu_time{};

  nvbench::int64_t m_num_trials{};

  nvbench::state &m_state;
};

template <typename KernelLauncher>
struct measure_hot : public measure_hot_base
{
  measure_hot(nvbench::state &state, KernelLauncher &kernel_launcher)
      : measure_hot_base(state)
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
    m_cuda_timer.start(m_launch.get_stream());
    this->launch_kernel();
    m_cuda_timer.stop(m_launch.get_stream());
  }

  void run_trials()
  {
    // Use warmup results to estimate the number of iterations to run.
    // The .95 factor here pads the batch_size a bit to avoid needing a second
    // batch due to noise.
    const auto time_estimate = m_cuda_timer.get_duration() * 0.95;
    auto batch_size = static_cast<nvbench::int64_t>(m_min_time / time_estimate);
    batch_size      = std::max(batch_size, nvbench::int64_t{1});

    do
    {
      m_cuda_timer.start(m_launch.get_stream());
      m_cpu_timer.start();
      for (nvbench::int64_t i = 0; i < batch_size; ++i)
      {
        this->launch_kernel();
      }
      m_cuda_timer.stop(m_launch.get_stream());
      NVBENCH_CUDA_CALL(cudaStreamSynchronize(m_launch.get_stream()));
      m_cpu_timer.stop();

      m_cuda_time += m_cuda_timer.get_duration();
      m_cpu_time += m_cpu_timer.get_duration();
      m_num_trials += batch_size;

      // Predict number of remaining iterations:
      batch_size = (m_min_time - m_cuda_time) / (m_cuda_time / m_num_trials);
    } while (batch_size > 0);
  }

  __forceinline__ void launch_kernel() { m_kernel_launcher(m_launch); }

  KernelLauncher &m_kernel_launcher;
};

} // namespace detail
} // namespace nvbench
