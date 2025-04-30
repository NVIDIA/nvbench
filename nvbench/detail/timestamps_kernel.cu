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

#include <nvbench/cuda_call.cuh>
#include <nvbench/cuda_stream.cuh>
#include <nvbench/detail/timestamps_kernel.cuh>
#include <nvbench/types.cuh>

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace
{

__global__ void get_timestamps_kernel(nvbench::uint64_t *global_timestamp,
                                      nvbench::uint64_t *sm0_timestamp)
{
  nvbench::uint32_t smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  if (smid == 0)
  {
    nvbench::uint64_t gts, lts;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gts));
    lts = clock64();

    *global_timestamp = gts;
    *sm0_timestamp    = lts;
  }
}

} // namespace

namespace nvbench::detail
{

timestamps_kernel::timestamps_kernel()
{
  NVBENCH_CUDA_CALL(
    cudaHostRegister(&m_host_timestamps, sizeof(nvbench::uint64_t) * 2, cudaHostRegisterMapped));
  NVBENCH_CUDA_CALL(cudaHostGetDevicePointer(&m_device_timestamps, &m_host_timestamps, 0));
}

timestamps_kernel::~timestamps_kernel()
{
  NVBENCH_CUDA_CALL_NOEXCEPT(cudaHostUnregister(&m_host_timestamps));
}

void timestamps_kernel::record(const nvbench::cuda_stream &stream)
{
  m_host_timestamps[0] = 0;
  m_host_timestamps[1] = 0;

  int device_id = 0;
  int num_sms   = 0;

  NVBENCH_CUDA_CALL(cudaGetDevice(&device_id));
  NVBENCH_CUDA_CALL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_id));

  get_timestamps_kernel<<<static_cast<unsigned int>(num_sms), 1, 0, stream.get_stream()>>>(
    m_device_timestamps,
    m_device_timestamps + 1);
}

} // namespace nvbench::detail
