#include <nvbench/cuda_call.cuh>

#include <fmt/format.h>

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace nvbench
{

namespace cuda_call
{

void throw_error(const std::string &filename,
                 std::size_t lineno,
                 const std::string &command,
                 cudaError_t error_code)
{
  throw std::runtime_error(fmt::format("{}:{}: Cuda API call returned error: "
                                       "{}: {}\nCommand: '{}'",
                                       filename,
                                       lineno,
                                       cudaGetErrorName(error_code),
                                       cudaGetErrorString(error_code),
                                       command));
}

void exit_error(const std::string &filename,
                std::size_t lineno,
                const std::string &command,
                cudaError_t error_code)
{
  fmt::print(stderr,
             "{}:{}: Cuda API call returned error: {}: {}\nCommand: '{}'",
             filename,
             lineno,
             cudaGetErrorName(error_code),
             cudaGetErrorString(error_code),
             command);
  std::exit(EXIT_FAILURE);
}

} // namespace cuda_call

} // namespace nvbench
