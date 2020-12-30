#pragma once

#include <cuda_runtime_api.h>

#include <string>

#define NVBENCH_CUDA_CALL(call)                                                \
  do                                                                           \
  {                                                                            \
    const cudaError_t nvbench_cuda_call_error = call;                          \
    if (nvbench_cuda_call_error != cudaSuccess)                                \
    {                                                                          \
      nvbench::cuda_call::throw_error(__FILE__,                                \
                                      __LINE__,                                \
                                      #call,                                   \
                                      nvbench_cuda_call_error);                \
    }                                                                          \
  } while (false)

namespace nvbench
{
namespace cuda_call
{

void throw_error(const std::string &filename,
                 std::size_t lineno,
                 const std::string &call,
                 cudaError_t error);

} // namespace cuda_call
} // namespace nvbench
