#pragma once

#include <nvbench/blocking_kernel.cuh>
#include <nvbench/cpu_timer.cuh>
#include <nvbench/cuda_call.cuh>
#include <nvbench/cuda_timer.cuh>
#include <nvbench/launch.cuh>
#include <nvbench/state.cuh>

#include <cuda_runtime.h>

#include <utility>

namespace nvbench::detail
{

// non-templated code goes here:
struct measure_hot_base
{
  explicit measure_hot_base(nvbench::state &exec_state);
  measure_hot_base(const measure_hot_base &) = delete;
  measure_hot_base(measure_hot_base &&)      = delete;
  measure_hot_base &operator=(const measure_hot_base &) = delete;
  measure_hot_base &operator=(measure_hot_base &&) = delete;

protected:
  void check();

  void initialize()
  {
    m_total_cpu_time    = 0.;
    m_total_cuda_time   = 0.;
    m_total_samples     = 0;
    m_max_time_exceeded = false;
  }

  void generate_summaries();

  nvbench::state &m_state;

  nvbench::launch m_launch;
  nvbench::cuda_timer m_cuda_timer;
  nvbench::cpu_timer m_cpu_timer;
  nvbench::cpu_timer m_timeout_timer;

  nvbench::int64_t m_min_samples{};
  nvbench::float64_t m_min_time{};
  nvbench::float64_t m_timeout{};

  nvbench::int64_t m_total_samples{};
  nvbench::float64_t m_total_cuda_time{};
  nvbench::float64_t m_total_cpu_time{};

  bool m_max_time_exceeded{false};
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
    this->check();
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
    m_timeout_timer.start();

    // Use warmup results to estimate the number of iterations to run.
    // The .95 factor here pads the batch_size a bit to avoid needing a second
    // batch due to noise.
    const auto time_estimate = m_cuda_timer.get_duration() * 0.95;
    auto batch_size = static_cast<nvbench::int64_t>(m_min_time / time_estimate);

    nvbench::blocking_kernel blocker;

    do
    {
      batch_size = std::max(batch_size, nvbench::int64_t{1});

      // Block stream until some work is queued.
      // Limit the number of kernel executions while blocked to prevent
      // deadlocks. See warnings on blocking_kernel.
      const auto blocked_launches   = std::min(batch_size, nvbench::int64_t{2});
      const auto unblocked_launches = batch_size - blocked_launches;

      blocker.block(m_launch.get_stream());
      m_cuda_timer.start(m_launch.get_stream());

      for (nvbench::int64_t i = 0; i < blocked_launches; ++i)
      {
        // If your benchmark deadlocks in the next launch, reduce the size of
        // blocked_launches. See note above.
        this->launch_kernel();
      }

      m_cpu_timer.start();
      blocker.unblock(); // Start executing earlier launches

      for (nvbench::int64_t i = 0; i < unblocked_launches; ++i)
      {
        this->launch_kernel();
      }

      m_cuda_timer.stop(m_launch.get_stream());
      NVBENCH_CUDA_CALL(cudaStreamSynchronize(m_launch.get_stream()));
      m_cpu_timer.stop();

      m_total_cpu_time += m_cpu_timer.get_duration();
      m_total_cuda_time += m_cuda_timer.get_duration();
      m_total_samples += batch_size;

      // Predict number of remaining iterations:
      batch_size = (m_min_time - m_total_cuda_time) /
                   (m_total_cuda_time / m_total_samples);

      m_timeout_timer.stop();
      const auto total_time = m_timeout_timer.get_duration();

      if (m_total_cuda_time > m_min_time && // min time okay
          m_total_samples > m_min_samples)  // min samples okay
      {
        break; // Stop iterating
      }

      if (m_total_cuda_time > m_timeout)
      {
        m_max_time_exceeded = true;
        break;
      }
    } while (true);
  }

  __forceinline__ void launch_kernel() { m_kernel_launcher(m_launch); }

  KernelLauncher &m_kernel_launcher;
};

} // namespace nvbench::detail
