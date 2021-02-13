#pragma once

#include <cuda_runtime_api.h>

#include <string>

/// Throws a std::runtime_error if `call` doesn't return `cudaSuccess`.
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

/// Terminates process with failure status if `call` doesn't return
/// `cudaSuccess`.
#define NVBENCH_CUDA_CALL_NOEXCEPT(call)                                       \
  do                                                                           \
  {                                                                            \
    const cudaError_t nvbench_cuda_call_error = call;                          \
    if (nvbench_cuda_call_error != cudaSuccess)                                \
    {                                                                          \
      nvbench::cuda_call::exit_error(__FILE__,                                 \
                                     __LINE__,                                 \
                                     #call,                                    \
                                     nvbench_cuda_call_error);                 \
    }                                                                          \
  } while (false)

namespace nvbench::cuda_call
{

void throw_error(const std::string &filename,
                 std::size_t lineno,
                 const std::string &call,
                 cudaError_t error);

void exit_error(const std::string &filename,
                std::size_t lineno,
                const std::string &command,
                cudaError_t error);

} // namespace nvbench::cuda_call
