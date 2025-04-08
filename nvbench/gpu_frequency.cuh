/*
 *  Copyright 2025 NVIDIA Corporation
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

#include <nvbench/timestamps_kernel.cuh>
#include <nvbench/types.cuh>

namespace nvbench
{

struct cuda_stream;

struct gpu_frequency
{
  __forceinline__ gpu_frequency() = default;

  // move-only
  gpu_frequency(const gpu_frequency &)            = delete;
  gpu_frequency(gpu_frequency &&)                 = default;
  gpu_frequency &operator=(const gpu_frequency &) = delete;
  gpu_frequency &operator=(gpu_frequency &&)      = default;

  __forceinline__ void start(const nvbench::cuda_stream &stream) { m_start.record(stream); }

  __forceinline__ void stop(const nvbench::cuda_stream &stream) { m_stop.record(stream); }

  [[nodiscard]] bool has_throttled(size_t peak_sm_clock_rate_hz,
                                   nvbench::float32_t throttle_threshold);

  [[nodiscard]] nvbench::float32_t get_clock_frequency();

private:
  nvbench::timestamps_kernel m_start;
  nvbench::timestamps_kernel m_stop;
};

} // namespace nvbench
