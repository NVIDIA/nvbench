#include <nvbench/cuda_call.cuh>

#include <cuda_runtime_api.h>

namespace nvbench
{

// RAII wrapper for a cudaStream_t.
struct cuda_stream
{
  cuda_stream() { NVBENCH_CUDA_CALL(cudaStreamCreate(&m_stream)); }
  ~cuda_stream() { NVBENCH_CUDA_CALL(cudaStreamDestroy(m_stream)); }

  // move-only
  cuda_stream(const cuda_stream &) = delete;
  cuda_stream(cuda_stream &&)      = default;
  cuda_stream &operator=(const cuda_stream &) = delete;
  cuda_stream &operator=(cuda_stream &&) = default;

  operator cudaStream_t() { return m_stream; }

private:
  cudaStream_t m_stream;
};

} // namespace nvbench
