// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <nvbench/config.cuh>

#if defined(NVBENCH_IMPLICIT_SYSTEM_HEADER_GCC)
#pragma GCC system_header
#elif defined(NVBENCH_IMPLICIT_SYSTEM_HEADER_CLANG)
#pragma clang system_header
#elif defined(NVBENCH_IMPLICIT_SYSTEM_HEADER_MSVC)
#pragma system_header
#endif

namespace nvbench::detail
{

struct measure_cold_launch_timer_config
{
  bool disable_blocking_kernel = false;
  bool run_once                = false;
  bool check_throttling        = true;
};

template <typename Measure>
struct measure_cold_launch_timer_core
{
private:
  void cleanup_noexcept() noexcept
  {
    const bool sync_armed = m_stream_unblock_armed || m_cuda_timer_started ||
                            m_gpu_frequency_cleanup_armed;

    if (m_stream_unblock_armed)
    {
      m_measure.unblock_stream_noexcept();
      m_stream_unblock_armed = false;
    }
    if (sync_armed)
    {
      (void)m_measure.sync_stream_noexcept();
    }
    if (m_profiler_started)
    {
      (void)m_measure.profiler_stop_noexcept();
      m_profiler_started = false;
    }
    if (m_cpu_timer_started)
    {
      m_measure.cpu_timer_stop_noexcept();
      m_cpu_timer_started = false;
    }

    m_cuda_timer_started          = false;
    m_gpu_frequency_cleanup_armed = false;
  }

  struct cleanup_guard
  {
    explicit cleanup_guard(measure_cold_launch_timer_core &timer)
        : m_timer{timer}
    {}

    cleanup_guard(const cleanup_guard &)            = delete;
    cleanup_guard(cleanup_guard &&)                 = delete;
    cleanup_guard &operator=(const cleanup_guard &) = delete;
    cleanup_guard &operator=(cleanup_guard &&)      = delete;

    ~cleanup_guard() noexcept
    {
      if (m_active)
      {
        m_timer.cleanup_noexcept();
      }
    }

    void release() noexcept { m_active = false; }

  private:
    measure_cold_launch_timer_core &m_timer;
    bool m_active{true};
  };

public:
  explicit measure_cold_launch_timer_core(Measure &measure, measure_cold_launch_timer_config config)
      : m_measure{measure}
      , m_disable_blocking_kernel{config.disable_blocking_kernel}
      , m_run_once{config.run_once}
      , m_check_throttling{config.check_throttling}
  {}

  measure_cold_launch_timer_core(const measure_cold_launch_timer_core &)            = delete;
  measure_cold_launch_timer_core(measure_cold_launch_timer_core &&)                 = delete;
  measure_cold_launch_timer_core &operator=(const measure_cold_launch_timer_core &) = delete;
  measure_cold_launch_timer_core &operator=(measure_cold_launch_timer_core &&)      = delete;

  ~measure_cold_launch_timer_core() noexcept { this->cleanup_noexcept(); }

  void start()
  {
    cleanup_guard cleanup{*this};

    m_measure.flush_device_l2();
    m_measure.sync_stream();

    // Start CPU timer irrespective of use of blocking kernel.
    // Ref: https://github.com/NVIDIA/nvbench/issues/249
    m_measure.cpu_timer_start();
    m_cpu_timer_started = true;

    if (!m_disable_blocking_kernel)
    {
      // Arm cleanup before queueing the blocking kernel. If block_stream throws
      // after queueing work, cleanup_noexcept must still unblock the stream.
      m_stream_unblock_armed = true;
      m_measure.block_stream();
    }
    if (m_check_throttling)
    {
      // Arm cleanup before queueing timestamp work. If gpu_frequency_start
      // throws after queueing work, cleanup_noexcept must still sync the stream.
      m_gpu_frequency_cleanup_armed = true;
      m_measure.gpu_frequency_start();
    }
    if (m_run_once)
    {
      m_measure.profiler_start();
      m_profiler_started = true;
    }
    m_measure.cuda_timer_start();
    m_cuda_timer_started = true;

    cleanup.release();
  }

  void stop()
  {
    cleanup_guard cleanup{*this};

    if (m_cuda_timer_started)
    {
      m_measure.cuda_timer_stop();
      m_cuda_timer_started = false;
    }
    if (m_gpu_frequency_cleanup_armed)
    {
      m_measure.gpu_frequency_stop();
      m_gpu_frequency_cleanup_armed = false;
    }
    if (m_stream_unblock_armed)
    {
      m_measure.unblock_stream();
      m_stream_unblock_armed = false;
    }
    m_measure.sync_stream();
    if (m_profiler_started)
    {
      m_measure.profiler_stop();
      m_profiler_started = false;
    }
    if (m_cpu_timer_started)
    {
      m_measure.cpu_timer_stop();
      m_cpu_timer_started = false;
    }

    cleanup.release();
  }

private:
  Measure &m_measure;
  bool m_disable_blocking_kernel     = false;
  bool m_run_once                    = false;
  bool m_check_throttling            = true;
  bool m_cpu_timer_started           = false;
  bool m_stream_unblock_armed        = false;
  bool m_gpu_frequency_cleanup_armed = false;
  bool m_profiler_started            = false;
  bool m_cuda_timer_started          = false;
};

} // namespace nvbench::detail
