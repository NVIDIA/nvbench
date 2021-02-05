#include <nvbench/device_info.cuh>

#include <nvbench/cuda_call.cuh>
#include <nvbench/detail/device_scope.cuh>

#include <cuda_runtime_api.h>

namespace nvbench
{

device_info::memory_info device_info::get_global_memory_usage() const
{
  nvbench::detail::device_scope _{m_id};

  memory_info result{};
  NVBENCH_CUDA_CALL(cudaMemGetInfo(&result.bytes_free, &result.bytes_total));
  return result;
}

device_info::device_info(int id)
    : m_id{id}
    , m_prop{}
{
  NVBENCH_CUDA_CALL(cudaGetDeviceProperties(&m_prop, m_id));
}

} // namespace nvbench
