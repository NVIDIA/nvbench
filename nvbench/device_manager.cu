#include <nvbench/device_manager.cuh>

#include <nvbench/cuda_call.cuh>
#include <nvbench/detail/device_scope.cuh>

#include <cuda_runtime_api.h>

namespace nvbench
{

device_manager &device_manager::get()
{
  static device_manager the_manager;
  return the_manager;
}

device_manager::device_manager()
{
  int num_devs{};
  NVBENCH_CUDA_CALL(cudaGetDeviceCount(&num_devs));
  m_devices.reserve(static_cast<std::size_t>(num_devs));

  for (int i = 0; i < num_devs; ++i)
  {
    m_devices.emplace_back(i);
  }
}

} // namespace nvbench
