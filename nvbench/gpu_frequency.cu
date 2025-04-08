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

#include <nvbench/gpu_frequency.cuh>

#include <iostream>

#include "nvbench/cuda_call.cuh"

namespace nvbench
{

struct cuda_stream;

nvbench::float32_t gpu_frequency::get_clock_frequency()
{
  nvbench::uint64_t elapsed_ns     = m_stop.m_host_timestamps[0] - m_start.m_host_timestamps[0];
  nvbench::uint64_t elapsed_clocks = m_stop.m_host_timestamps[1] - m_start.m_host_timestamps[1];
  nvbench::float32_t clock_rate    = float(elapsed_clocks) / float(elapsed_ns) * 1000000.f;
  return clock_rate;
}

bool gpu_frequency::has_throttled()
{
  int deviceId     = 0;
  int maxClockRate = 0;
  NVBENCH_CUDA_CALL_NOEXCEPT(cudaGetDevice(&deviceId));
  NVBENCH_CUDA_CALL_NOEXCEPT(cudaDeviceGetAttribute(&maxClockRate, cudaDevAttrClockRate, deviceId));
  float throttleThreshold = static_cast<float>(maxClockRate) * 0.8f; // TODO extract into parameter

  if (get_clock_frequency() < throttleThreshold)
  {
    return true;
  }

  return false;
}

} // namespace nvbench
