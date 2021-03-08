/*
 *  Copyright 2021 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 with the LLVM exception
 *  (the "License"); you may not use this file except in compliance with
 *  the License.
 *
 *  You may obtain a copy of the License at
 *
 *      http://llvm.org/foundation/relicensing/LICENSE.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

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
