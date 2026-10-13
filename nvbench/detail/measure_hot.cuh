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

#include <nvbench/config.cuh>

#if defined(NVBENCH_IMPLICIT_SYSTEM_HEADER_GCC)
#pragma GCC system_header
#elif defined(NVBENCH_IMPLICIT_SYSTEM_HEADER_CLANG)
#pragma clang system_header
#elif defined(NVBENCH_IMPLICIT_SYSTEM_HEADER_MSVC)
#pragma system_header
#endif

#include <nvbench/blocking_kernel.cuh>
#include <nvbench/cpu_timer.cuh>
#include <nvbench/cuda_call.cuh>
#include <nvbench/cuda_timer.cuh>
#include <nvbench/detail/persisting_l2_cache_reset_fwd.cuh>
#include <nvbench/detail/stream_cleanup_guard.cuh>
#include <nvbench/exec_tag.cuh>
#include <nvbench/launch.cuh>

#include <cuda_runtime.h>

#include <algorithm>

namespace nvbench
{

struct state;

namespace detail
{

// non-templated code goes here to keep instantiation cost down:
struct measure_hot_base
{
  explicit measure_hot_base(nvbench::state &exec_state);
  ~measure_hot_base();
  measure_hot_base(const measure_hot_base &)            = delete;
  measure_hot_base(measure_hot_base &&)                 = delete;
  measure_hot_base &operator=(const measure_hot_base &) = delete;
  measure_hot_base &operator=(measure_hot_base &&)      = delete;

protected:
  friend struct nvbench::detail::stream_cleanup_guard<measure_hot_base>;

  void check();

  void initialize()
  {
    m_total_cuda_time   = 0.;
    m_total_samples     = 0;
    m_max_time_exceeded = false;
  }

  void generate_summaries();

  void check_skip_time(nvbench::float64_t warmup_time);

  void block_stream();
  void initialize_persisting_l2_cache_disable();
  void reset_persisting_l2_cache();
  void restore_persisting_l2_cache();

  static nvbench::int64_t predict_cuda_batch_size(nvbench::float64_t target_time,
                                                  nvbench::float64_t time_estimate,
                                                  nvbench::int64_t minimum_batch_size,
                                                  nvbench::int64_t fallback_on_invalid_prediction);
  static nvbench::int64_t predict_timeout_batch_cap(nvbench::float64_t target_time,
                                                    nvbench::float64_t time_estimate,
                                                    nvbench::int64_t cuda_batch_size);
  static nvbench::int64_t grow_batch_size(nvbench::int64_t batch_size,
                                          nvbench::int64_t minimum_batch_size);

  static constexpr nvbench::int64_t minimum_hot_batch_size = 4;

  __forceinline__ void unblock_stream() { m_blocker.unblock(); }
  __forceinline__ void unblock_stream_noexcept() noexcept { m_blocker.unblock_noexcept(); }

  __forceinline__ cudaError_t sync_stream_noexcept() const noexcept
  {
    return cudaStreamSynchronize(m_launch.get_stream());
  }

  __forceinline__ void sync_stream() const { NVBENCH_CUDA_CALL(this->sync_stream_noexcept()); }

  nvbench::state &m_state;

  nvbench::launch m_launch;
  nvbench::cuda_timer m_cuda_timer;
  nvbench::cpu_timer m_walltime_timer;
  nvbench::blocking_kernel m_blocker;
  nvbench::detail::persisting_l2_cache_disable_ptr m_persisting_l2_cache_disable{};

  nvbench::int64_t m_min_samples{};
  nvbench::float64_t m_batch_target_time{};

  nvbench::float64_t m_skip_time{};
  nvbench::float64_t m_timeout{};

  nvbench::int64_t m_total_samples{};
  nvbench::float64_t m_total_cuda_time{};

  bool m_disable_blocking_kernel{false};
  bool m_disable_persisting_l2_cache{false};
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
    this->initialize_persisting_l2_cache_disable();

    try
    {
      this->run_warmup();
      this->run_trials();
      this->restore_persisting_l2_cache();
    }
    catch (...)
    {
      this->restore_persisting_l2_cache();
      throw;
    }

    this->generate_summaries();
  }

private:
  // Run the kernel once, measuring the GPU time. If under skip_time, skip the
  // measurement.
  void run_warmup()
  {
    nvbench::detail::stream_cleanup_guard<measure_hot_base> cleanup{*this};

    this->reset_persisting_l2_cache();
    m_walltime_timer.start();
    {
      m_cuda_timer.start(m_launch.get_stream());
      this->launch_kernel();
      m_cuda_timer.stop(m_launch.get_stream());

      this->sync_stream();
    }
    // get wall-clock estimate of launch execution
    m_walltime_timer.stop();
    cleanup.release();

    this->check_skip_time(m_cuda_timer.get_duration());
  }

  void run_trials()
  {
    const auto wallclock_time_initial_estimate = m_walltime_timer.get_duration();
    const auto cuda_time_initial_estimate      = m_cuda_timer.get_duration();

    m_walltime_timer.start();

    // Use warmup results to estimate the number of iterations to run.
    // The .95 factor here pads the batch_size a bit to avoid needing a second
    // batch due to noise.
    const auto hot_batch_size_floor = std::min(std::max(m_min_samples, nvbench::int64_t{1}),
                                               minimum_hot_batch_size);
    const auto time_estimate        = cuda_time_initial_estimate * 0.95;
    auto batch_size                 = this->predict_cuda_batch_size(m_batch_target_time,
                                                                    time_estimate,
                                                                    hot_batch_size_floor,
                                                                    m_min_samples);
    auto timeout_batch_size =
      this->predict_timeout_batch_cap(m_timeout, wallclock_time_initial_estimate, batch_size);

    do
    {
      batch_size = std::min(batch_size, timeout_batch_size);

      nvbench::detail::stream_cleanup_guard<measure_hot_base> cleanup{*this};

      this->reset_persisting_l2_cache();
      if (!m_disable_blocking_kernel)
      {
        // Block stream until some work is queued.
        // Limit the number of kernel executions while blocked to prevent
        // deadlocks. See warnings on blocking_kernel.
        const auto blocked_launches   = std::min(batch_size, nvbench::int64_t{2});
        const auto unblocked_launches = batch_size - blocked_launches;

        cleanup.block_stream();
        m_cuda_timer.start(m_launch.get_stream());

        for (nvbench::int64_t i = 0; i < blocked_launches; ++i)
        {
          // If your benchmark deadlocks in the next launch, reduce the size of
          // blocked_launches. See note above.
          this->launch_kernel();
        }

        cleanup.unblock(); // Start executing earlier launches

        for (nvbench::int64_t i = 0; i < unblocked_launches; ++i)
        {
          this->launch_kernel();
        }
      }
      else
      {
        m_cuda_timer.start(m_launch.get_stream());

        for (nvbench::int64_t i = 0; i < batch_size; ++i)
        {
          this->launch_kernel();
        }
      }

      m_cuda_timer.stop(m_launch.get_stream());
      this->sync_stream();
      cleanup.release();

      m_total_cuda_time += m_cuda_timer.get_duration();
      m_total_samples += batch_size;

      if (m_total_cuda_time >= m_batch_target_time && // batch target time okay
          m_total_samples >= m_min_samples)           // min samples okay
      {
        break; // Stop iterating
      }

      const auto sample_count = static_cast<nvbench::float64_t>(m_total_samples);

      // Predict number of remaining iterations based on cuda-time budget
      const auto remaining_time               = m_batch_target_time - m_total_cuda_time;
      const auto time_per_sample              = m_total_cuda_time / sample_count;
      const auto remaining_samples_to_minimum = std::max(m_min_samples - m_total_samples,
                                                         nvbench::int64_t{1});
      const auto batch_target_time_satisfied  = remaining_time <= nvbench::float64_t{0};
      const auto fallback_size_on_invalid_cuda_prediction = batch_target_time_satisfied
                                                              ? remaining_samples_to_minimum
                                                              : m_min_samples;
      batch_size =
        this->predict_cuda_batch_size(remaining_time,
                                      time_per_sample,
                                      this->grow_batch_size(batch_size, hot_batch_size_floor),
                                      fallback_size_on_invalid_cuda_prediction);

      m_walltime_timer.stop();
      const auto total_walltime = m_walltime_timer.get_duration();
      if (total_walltime > m_timeout)
      {
        m_max_time_exceeded = true;
        break;
      }

      // Predict number of remaining iterations based on timeout budget.
      const auto remaining_walltime  = m_timeout - total_walltime;
      const auto walltime_per_sample = total_walltime / sample_count;
      timeout_batch_size =
        this->predict_timeout_batch_cap(remaining_walltime, walltime_per_sample, batch_size);

    } while (true);

    m_walltime_timer.stop();
  }

  __forceinline__ void launch_kernel() { m_kernel_launcher(m_launch); }

  KernelLauncher &m_kernel_launcher;
};

} // namespace detail
} // namespace nvbench
