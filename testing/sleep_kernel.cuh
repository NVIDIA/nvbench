#pragma once

#include <cuda/std/chrono>

#include <cuda_runtime.h>

__global__ void sleep_kernel(double seconds)
{
  const auto start = cuda::std::chrono::high_resolution_clock::now();
  const auto ns    = cuda::std::chrono::nanoseconds(
    static_cast<nvbench::int64_t>(seconds * 1000 * 1000 * 1000));
  const auto finish = start + ns;

  auto now = cuda::std::chrono::high_resolution_clock::now();
  while (now < finish)
  {
    now = cuda::std::chrono::high_resolution_clock::now();
  }
}
