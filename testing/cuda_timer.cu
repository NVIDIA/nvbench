#include <nvbench/cuda_timer.cuh>

#include <nvbench/cuda_stream.cuh>
#include <nvbench/types.cuh>

#include "sleep_kernel.cuh"
#include "test_asserts.cuh"

#include <fmt/format.h>

void test_basic(cudaStream_t time_stream,
                cudaStream_t exec_stream,
                bool expected)
{
  nvbench::cuda_timer timer;

  NVBENCH_CUDA_CALL(cudaDeviceSynchronize());

  timer.start(time_stream);
  sleep_kernel<<<1, 1, 0, exec_stream>>>(0.25);
  timer.stop(time_stream);

  NVBENCH_CUDA_CALL(cudaDeviceSynchronize());
  const bool captured = timer.get_duration() > 0.25;
  ASSERT_MSG(captured == expected,
             "Unexpected result from timer: {} seconds (expected {})",
             timer.get_duration(),
             (expected ? "> 0.25s" : "< 0.25s"));
}

void test_basic()
{
  nvbench::cuda_stream stream1;
  nvbench::cuda_stream stream2;

  test_basic(stream1, stream1, true);
  test_basic(stream1, stream2, false);
}

int main() { test_basic(); }
