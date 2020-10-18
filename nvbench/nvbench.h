// TODO license. I'd like to make this BSD 3-clause for simplicity.
// TODO Need a way to select device at runtime.
// TODO Need to customize reported times more: Wall *and/or* CPU *and/or* GPU
// TODO Implement GPU-time only benchmarks.
// TODO Scale max range to device limits via BENCHMARK_APPLY macros

#pragma once

#include <benchmark/benchmark.h>

#include <cuda_runtime_api.h>

#define CUDA_CALL(call)                                                        \
  do                                                                           \
  {                                                                            \
    cudaError err = call;                                                      \
    if (cudaSuccess != err)                                                    \
    {                                                                          \
      fprintf(stderr,                                                          \
              "CUDA error in file '%s' in line %i : %s.\n",                    \
              __FILE__,                                                        \
              __LINE__,                                                        \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

namespace nvbench
{

// Modified from existing thrust benchmark code:
class cuda_timer
{
  cudaEvent_t m_start;
  cudaEvent_t m_stop;
  cudaStream_t m_stream;

public:
  cuda_timer(cudaStream_t stream = 0)
      : m_stream(stream)
  {
    CUDA_CALL(cudaEventCreate(&m_start));
    CUDA_CALL(cudaEventCreate(&m_stop));
  }

  ~cuda_timer()
  {
    CUDA_CALL(cudaEventDestroy(m_start));
    CUDA_CALL(cudaEventDestroy(m_stop));
  }

  void start() { CUDA_CALL(cudaEventRecord(m_start, m_stream)); }

  void stop() { CUDA_CALL(cudaEventRecord(m_stop, m_stream)); }

  bool ready()
  {
    cudaError_t const state = cudaEventQuery(m_stop);
    if (state == cudaErrorNotReady)
    {
      return false;
    }
    CUDA_CALL(state);
    return true;
  }

  double milliseconds_elapsed()
  {
    CUDA_CALL(cudaEventSynchronize(m_start));
    CUDA_CALL(cudaEventSynchronize(m_stop));
    float elapsed_time;
    CUDA_CALL(cudaEventElapsedTime(&elapsed_time, m_start, m_stop));
    return elapsed_time;
  }

  double seconds_elapsed() { return milliseconds_elapsed() / 1000.0; }
};

} // namespace nvbench
