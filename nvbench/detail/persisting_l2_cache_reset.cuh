/*
 *  Copyright 2026 NVIDIA Corporation
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

#include <nvbench/cuda_call.cuh>
#include <nvbench/detail/device_scope.cuh>
#include <nvbench/detail/throw.cuh>
#include <nvbench/device_info.cuh>

#include <cuda_runtime_api.h>

#include <optional>
#include <stdexcept>

namespace nvbench::detail
{

struct persisting_l2_cache_reset
{
  explicit persisting_l2_cache_reset(int device_id)
      : m_device_id{device_id}
  {}

  void operator()() const
  {
    nvbench::detail::device_scope scope{m_device_id};

    int max_persisting_l2_cache_size{};
    NVBENCH_CUDA_CALL(cudaDeviceGetAttribute(&max_persisting_l2_cache_size,
                                             cudaDevAttrMaxPersistingL2CacheSize,
                                             m_device_id));
    if (max_persisting_l2_cache_size == 0)
    {
      return;
    }

    NVBENCH_CUDA_CALL(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0));
    NVBENCH_CUDA_CALL(cudaCtxResetPersistingL2Cache());
  }

private:
  int m_device_id{};
};

inline void reset_persisting_l2_cache_if_requested(bool requested,
                                                   const std::optional<nvbench::device_info> &device)
{
  if (!requested)
  {
    return;
  }

  if (!device)
  {
    NVBENCH_THROW(std::runtime_error, "{}", "Device required to disable persisting L2 cache.");
  }

  nvbench::detail::persisting_l2_cache_reset{device->get_id()}();
}

} // namespace nvbench::detail
