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

#include <nvbench/device_info.cuh>

#include <nvbench/cuda_call.cuh>
#include <nvbench/detail/device_scope.cuh>

#include <cuda_runtime_api.h>

namespace nvbench
{

device_info::memory_info device_info::get_global_memory_usage() const
{
  nvbench::detail::device_scope _{m_id};

  memory_info result{};
  NVBENCH_CUDA_CALL(cudaMemGetInfo(&result.bytes_free, &result.bytes_total));
  return result;
}

device_info::device_info(int id)
    : m_id{id}
    , m_prop{}
{
  NVBENCH_CUDA_CALL(cudaGetDeviceProperties(&m_prop, m_id));
}

} // namespace nvbench
