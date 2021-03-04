/*
 *  Copyright 2020 NVIDIA Corporation
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

#include <nvbench/blocking_kernel.cuh>

#include <nvbench/cuda_call.cuh>
#include <nvbench/cuda_stream.cuh>

#include <cuda_runtime.h>

#include <cstdlib>

namespace
{

__global__ void block_stream(const volatile int *flag)
{
  while (!(*flag))
  {}
}

} // namespace

namespace nvbench
{

blocking_kernel::blocking_kernel()
{
  NVBENCH_CUDA_CALL(
    cudaHostRegister(&m_host_flag, sizeof(m_host_flag), cudaHostRegisterMapped));
  NVBENCH_CUDA_CALL(cudaHostGetDevicePointer(&m_device_flag, &m_host_flag, 0));
}

blocking_kernel::~blocking_kernel()
{
  NVBENCH_CUDA_CALL_NOEXCEPT(cudaHostUnregister(&m_host_flag));
}

void blocking_kernel::block(const nvbench::cuda_stream &stream)
{
  m_host_flag = 0;
  block_stream<<<1, 1, 0, stream>>>(m_device_flag);
}

} // namespace nvbench
