// TODO license. I'd like to make this BSD 3-clause for simplicity.
// TODO Need a way to select device at runtime.

#pragma once

#include <benchmark/benchmark.h>

#include <cuda_runtime_api.h>

#define CUDA_CALL(call)                                                        \
  do {                                                                         \
    cudaError err = call;                                                      \
    if (cudaSuccess != err) {                                                  \
      fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n", __FILE__,  \
              __LINE__, cudaGetErrorString(err));                              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)


namespace nvbench
{

// Modified from existing thrust benchmark code:
class cuda_timer
{
  cudaEvent_t start_;
  cudaEvent_t stop_;

public:
  cuda_timer()
  {
    CUDA_CALL(cudaEventCreate(&start_));
    CUDA_CALL(cudaEventCreate(&stop_));
  }

  ~cuda_timer()
  {
    CUDA_CALL(cudaEventDestroy(start_));
    CUDA_CALL(cudaEventDestroy(stop_));
  }

  void start()
  {
    CUDA_CALL(cudaEventRecord(start_, 0));
  }

  void stop()
  {
    CUDA_CALL(cudaEventRecord(stop_, 0));
  }

  double milliseconds_elapsed()
  {
    CUDA_CALL(cudaEventSynchronize(start_));
    CUDA_CALL(cudaEventSynchronize(stop_));
    float elapsed_time;
    CUDA_CALL(cudaEventElapsedTime(&elapsed_time, start_, stop_));
    return elapsed_time;
  }

  double seconds_elapsed()
  {
    return milliseconds_elapsed() / 1000.0;
  }
};

}
