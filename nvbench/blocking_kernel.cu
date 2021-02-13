#include <nvbench/blocking_kernel.cuh>

#include <nvbench/cuda_call.cuh>
#include <nvbench/cuda_stream.cuh>
#include <nvbench/detail/throw.cuh>

#include <fmt/format.h>

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

void blocking_kernel::release()
{
  volatile int& flag = m_host_flag;
  flag = 1;
}

} // namespace nvbench
