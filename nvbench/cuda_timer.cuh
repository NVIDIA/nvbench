#pragma once

#include <nvbench/cuda_call.cuh>

#include <cuda_runtime_api.h>

namespace nvbench
{

struct cuda_timer
{
  cuda_timer()
  {
    NVBENCH_CUDA_CALL(cudaEventCreate(&m_start));
    NVBENCH_CUDA_CALL(cudaEventCreate(&m_stop));
  }

  ~cuda_timer()
  {
    NVBENCH_CUDA_CALL(cudaEventDestroy(m_start));
    NVBENCH_CUDA_CALL(cudaEventDestroy(m_stop));
  }

  // move-only
  cuda_timer(const cuda_timer &) = delete;
  cuda_timer(cuda_timer &&)      = default;
  cuda_timer &operator=(const cuda_timer &) = delete;
  cuda_timer &operator=(cuda_timer &&) = default;

  void start(cudaStream_t stream)
  {
    NVBENCH_CUDA_CALL(cudaEventRecord(m_start, stream));
  }

  void stop(cudaStream_t stream)
  {
    NVBENCH_CUDA_CALL(cudaEventRecord(m_stop, stream));
  }

  bool ready() const
  {
    const cudaError_t state = cudaEventQuery(m_stop);
    if (state == cudaErrorNotReady)
    {
      return false;
    }
    NVBENCH_CUDA_CALL(state);
    return true;
  }

  double get_duration() const
  {
    NVBENCH_CUDA_CALL(cudaEventSynchronize(m_stop));
    float elapsed_time;
    // According to docs, this is in ms with a resolution of ~0.5 microseconds.
    NVBENCH_CUDA_CALL(cudaEventElapsedTime(&elapsed_time, m_start, m_stop));
    return elapsed_time / 1000.0;
  }

private:
  cudaEvent_t m_start;
  cudaEvent_t m_stop;
};

} // namespace nvbench
