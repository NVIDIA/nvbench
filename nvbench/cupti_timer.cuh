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

#include <nvbench/types.cuh>

#include <cuda_runtime_api.h>

#include <optional>

namespace nvbench
{

/**
 * GPU timer based on the CUPTI Activity API.
 *
 * Measures the GPU busy time of the kernels, memcpys, and memsets between `start()` and `stop()`:
 * In constrast to CUDA events, kernel launch overhead and idle time frames between operations are
 * excluded. In addition, work queued outside of the window, such as blocking kernel or an L2 flush,
 * is ignored.
 */
struct cupti_timer
{
  cupti_timer();
  ~cupti_timer();

  cupti_timer(const cupti_timer &)            = delete;
  cupti_timer(cupti_timer &&)                 = delete;
  cupti_timer &operator=(const cupti_timer &) = delete;
  cupti_timer &operator=(cupti_timer &&)      = delete;

  void start(cudaStream_t stream);
  void stop(cudaStream_t stream);

  // In seconds. Synchronizes the device.
  [[nodiscard]] nvbench::float64_t get_duration() const;

private:
  nvbench::uint64_t m_id = 0;
  bool m_pushed          = false;
  // nullopt_t means the duration is not yet available.
  mutable std::optional<nvbench::float64_t> m_duration{};

  void pop_noexcept() noexcept;
};

} // namespace nvbench
