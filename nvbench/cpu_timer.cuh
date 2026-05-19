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

#include <nvbench/types.cuh>

#include <chrono>
#include <type_traits>

namespace nvbench
{

struct cpu_timer
{
  __forceinline__ cpu_timer() = default;

  // move-only
  cpu_timer(const cpu_timer &)            = delete;
  cpu_timer(cpu_timer &&)                 = default;
  cpu_timer &operator=(const cpu_timer &) = delete;
  cpu_timer &operator=(cpu_timer &&)      = default;

  __forceinline__ void start() noexcept { m_start = cpu_timer_clock::now(); }

  __forceinline__ void stop() noexcept { m_stop = cpu_timer_clock::now(); }

  // In seconds:
  [[nodiscard]] __forceinline__ nvbench::float64_t get_duration()
  {
    const auto duration = m_stop - m_start;
    const auto ns       = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    return static_cast<nvbench::float64_t>(ns) * (1e-9);
  }

private:
  // Use high_resolution_clock only when it is monotonic; otherwise fall back to
  // steady_clock to avoid negative elapsed times after system clock adjustments.
  using cpu_timer_clock = std::conditional_t<std::chrono::high_resolution_clock::is_steady,
                                             std::chrono::high_resolution_clock,
                                             std::chrono::steady_clock>;
  static_assert(cpu_timer_clock::is_steady, "cpu_timer requires a steady clock.");

  using time_point_t = cpu_timer_clock::time_point;

  time_point_t m_start;
  time_point_t m_stop;
};

} // namespace nvbench
