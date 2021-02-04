#pragma once

#include <nvbench/device_info.cuh>

#include <vector>

namespace nvbench
{

/**
 * Singleton class that caches CUDA device information.
 */
struct device_manager
{
  using device_info_vector = std::vector<nvbench::device_info>;

  /**
   * @return The singleton benchmark_manager instance.
   */
  [[nodiscard]] static device_manager &get();

  [[nodiscard]] int get_number_of_devices() const
  {
    return static_cast<int>(m_devices.size());
  }

  [[nodiscard]] const nvbench::device_info &get_device(int id)
  {
    return m_devices.at(id);
  }

  [[nodiscard]] const device_info_vector &get_devices() const
  {
    return m_devices;
  }

private:
  device_manager();

  device_info_vector m_devices;
};

} // namespace nvbench
