#include <nvbench/device_info.cuh>

#include <nvbench/cuda_call.cuh>
#include <nvbench/detail/device_scope.cuh>

#include <cuda_runtime_api.h>

#include <cstdint> // CHAR_BIT

namespace nvbench
{

device_info::memory_info device_info::get_global_memory_info() const
{
  nvbench::detail::device_scope _{m_id};

  memory_info result{};
  NVBENCH_CUDA_CALL(cudaMemGetInfo(&result.bytes_free, &result.bytes_total));
  return result;
}

std::size_t device_info::get_global_memory_bandwidth() const
{
  // Global memory bus is DDR:
  const std::size_t ddr_factor = 2;

  // prop is KHz, convert to Hz:
  const std::size_t memory_clock_rate_hz = m_prop.memoryClockRate * 1000;

  // Memory bus is specified in bits:
  const std::size_t bus_width = m_prop.memoryBusWidth / CHAR_BIT;

  return ddr_factor * memory_clock_rate_hz * bus_width;
}

device_info::device_info(int id)
  : m_id{id}
  , m_prop{}
{
  NVBENCH_CUDA_CALL(cudaGetDeviceProperties(&m_prop, m_id));
}

} // namespace nvbench
