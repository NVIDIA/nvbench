#pragma once

#include <nvbench/cuda_call.cuh>
#include <nvbench/detail/device_scope.cuh>

#include <cuda_runtime_api.h>

#include <string_view>
#include <utility>

namespace nvbench
{

namespace detail
{
int get_ptx_version(int);
}

struct device_info
{
  explicit device_info(int id);

  /// @return The device's id on the current system.
  [[nodiscard]] int get_id() const { return m_id; }

  /// @return The name of the device.
  [[nodiscard]] std::string_view get_name() const
  {
    return std::string_view(m_prop.name);
  }

  /// @return The SM version of the current device as (major*100) + (minor*10).
  [[nodiscard]] int get_sm_version() const
  {
    return m_prop.major * 100 + m_prop.minor * 10;
  }

  /// @return The PTX version of the current device
  [[nodiscard]] __forceinline__ int get_ptx_version() const
  {
    return detail::get_ptx_version(m_id);
  }

  /// @return The number of physical streaming multiprocessors on this device.
  [[nodiscard]] int get_number_of_sms() const
  {
    return m_prop.multiProcessorCount;
  }

  /// @return The maximum number of resident blocks per SM.
  [[nodiscard]] int get_max_blocks_per_sm() const
  {
    return m_prop.maxBlocksPerMultiProcessor;
  }

  /// @return The maximum number of resident threads per SM.
  [[nodiscard]] int get_max_threads_per_sm() const
  {
    return m_prop.maxThreadsPerMultiProcessor;
  }

  /// @return The maximum number of threads per block.
  [[nodiscard]] int get_max_threads_per_block() const
  {
    return m_prop.maxThreadsPerBlock;
  }

  /// @return The number of registers per SM.
  [[nodiscard]] int get_registers_per_sm() const
  {
    return m_prop.regsPerMultiprocessor;
  }

  /// @return The number of registers per block.
  [[nodiscard]] int get_registers_per_block() const
  {
    return m_prop.regsPerBlock;
  }

  struct memory_info
  {
    std::size_t bytes_free;
    std::size_t bytes_total;
  };

  /// @return The size and usage of this device's global memory.
  [[nodiscard]] memory_info get_global_memory_info() const;

  //// @return The combined read/write global memory bandwidth in bytes/sec.
  [[nodiscard]] std::size_t get_global_memory_bandwidth() const;

  /// @return The size of the L2 cache in bytes.
  [[nodiscard]] std::size_t get_l2_cache_size() const
  {
    return m_prop.l2CacheSize;
  }

  /// @return True if ECC is enabled on this device.
  [[nodiscard]] bool get_ecc_state() const { return m_prop.ECCEnabled; }

  /// @return A cached copy of the device's cudaDeviceProp.
  [[nodiscard]] const cudaDeviceProp &get_cuda_device_prop() const
  {
    return m_prop;
  }

private:
  int m_id;
  cudaDeviceProp m_prop;
};

// get_ptx_version implementation; this needs to stay in the header so it will
// pick up the downstream project's compilation settings.
namespace detail
{
// Templated to workaround ODR issues since __global__functions cannot be marked
// inline.
template <typename>
__global__ void noop_kernel()
{}

inline const auto noop_kernel_ptr = &noop_kernel<void>;

[[nodiscard]] inline int get_ptx_version(int dev_id)
{
  nvbench::detail::device_scope _{dev_id};
  cudaFuncAttributes attr{};
  NVBENCH_CUDA_CALL(
    cudaFuncGetAttributes(&attr, nvbench::detail::noop_kernel_ptr));
  return attr.ptxVersion * 10;
}
} // namespace detail

} // namespace nvbench
