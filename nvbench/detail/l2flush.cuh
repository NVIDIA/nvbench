#pragma once

#include <nvbench/cuda_call.cuh>

#include <cuda_runtime_api.h>

namespace nvbench::detail
{

struct l2flush
{
  __forceinline__ l2flush()
      : m_l2_buffer{nullptr}
  {
    int dev_id{};
    NVBENCH_CUDA_CALL(cudaGetDevice(&dev_id));
    NVBENCH_CUDA_CALL(
      cudaDeviceGetAttribute(&m_l2_size, cudaDevAttrL2CacheSize, dev_id));
    if (m_l2_size > 0)
    {
      NVBENCH_CUDA_CALL(cudaMalloc(&m_l2_buffer, m_l2_size));
    }
  }

  __forceinline__ ~l2flush()
  {
    if (m_l2_buffer)
    {
      NVBENCH_CUDA_CALL(cudaFree(m_l2_buffer));
    }
  }

  __forceinline__ void flush(cudaStream_t stream)
  {
    if (m_l2_size > 0)
    {
      NVBENCH_CUDA_CALL(cudaMemsetAsync(m_l2_buffer, 0, m_l2_size, stream));
    }
  }

private:
  int m_l2_size;
  int *m_l2_buffer;
};

} // namespace nvbench::detail
