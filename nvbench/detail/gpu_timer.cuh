// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
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

#include <nvbench/cuda_timer.cuh>
#include <nvbench/types.cuh>

#ifdef NVBENCH_HAS_CUPTI
#include <nvbench/cupti_timer.cuh>
#endif

#include <cuda_runtime_api.h>

#include <optional>

namespace nvbench::detail
{

// Measures GPU time with CUPTI when requested, and with CUDA events otherwise.
struct gpu_timer
{
  explicit gpu_timer([[maybe_unused]] bool use_cupti) noexcept
  {
#ifdef NVBENCH_HAS_CUPTI
    if (use_cupti)
    {
      m_cupti.emplace();
    }
#endif
  }

  [[nodiscard]] bool uses_cupti() const noexcept
  {
#ifdef NVBENCH_HAS_CUPTI
    return m_cupti.has_value();
#else
    return false;
#endif
  }

  __forceinline__ void start(cudaStream_t stream)
  {
#ifdef NVBENCH_HAS_CUPTI
    if (m_cupti)
    {
      m_cupti->start(stream);
      return;
    }
#endif
    m_events.start(stream);
  }

  __forceinline__ void stop(cudaStream_t stream)
  {
#ifdef NVBENCH_HAS_CUPTI
    if (m_cupti)
    {
      m_cupti->stop(stream);
      return;
    }
#endif
    m_events.stop(stream);
  }

  // In seconds:
  [[nodiscard]] __forceinline__ nvbench::float64_t get_duration() const
  {
#ifdef NVBENCH_HAS_CUPTI
    if (m_cupti)
    {
      return m_cupti->get_duration();
    }
#endif
    return m_events.get_duration();
  }

  [[nodiscard]] const char *description() const noexcept
  {
    return this->uses_cupti() ? "measured with CUPTI" : "measured with CUDA events";
  }

private:
  nvbench::cuda_timer m_events{};
#ifdef NVBENCH_HAS_CUPTI
  // nullopt_t means the duration is not yet available.
  std::optional<nvbench::cupti_timer> m_cupti{};
#endif
};

} // namespace nvbench::detail
