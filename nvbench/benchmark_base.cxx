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

#include <nvbench/benchmark_base.cuh>
#include <nvbench/criterion_manager.cuh>
#include <nvbench/detail/transform_reduce.cuh>

#include <algorithm>
#include <cstdint>

namespace nvbench
{

benchmark_base::~benchmark_base() = default;

std::unique_ptr<benchmark_base> benchmark_base::clone() const
{
  auto result = this->do_clone();

  // Do not copy states.
  result->m_name    = m_name;
  result->m_axes    = m_axes;
  result->m_devices = m_devices;

  result->m_printer = m_printer;

  result->m_is_cpu_only             = m_is_cpu_only;
  result->m_run_once                = m_run_once;
  result->m_disable_blocking_kernel = m_disable_blocking_kernel;

  result->m_min_samples = m_min_samples;

  result->m_skip_time = m_skip_time;
  result->m_timeout   = m_timeout;

  result->m_criterion_params        = m_criterion_params;
  result->m_throttle_threshold      = m_throttle_threshold;
  result->m_throttle_recovery_delay = m_throttle_recovery_delay;

  result->m_stopping_criterion = m_stopping_criterion;

  return result;
}

benchmark_base &benchmark_base::set_devices(std::vector<int> device_ids)
{
  std::vector<device_info> devices;
  devices.reserve(device_ids.size());
  for (int dev_id : device_ids)
  {
    devices.emplace_back(dev_id);
  }
  return this->set_devices(std::move(devices));
}

benchmark_base &benchmark_base::add_device(int device_id)
{
  return this->add_device(device_info{device_id});
}

std::size_t benchmark_base::get_config_count() const
{
  const std::size_t per_device_count = nvbench::detail::transform_reduce(
    m_axes.get_axes().cbegin(),
    m_axes.get_axes().cend(),
    std::size_t{1},
    std::multiplies<>{},
    [](const auto &axis_ptr) {
      if (const auto *type_axis_ptr = dynamic_cast<const nvbench::type_axis *>(axis_ptr.get());
          type_axis_ptr != nullptr)
      {
        return type_axis_ptr->get_active_count();
      }
      return axis_ptr->get_size();
    });

  // Devices will be empty for cpu-only benchmarks.
  return per_device_count * std::max(std::size_t(1), m_devices.size());
}

benchmark_base &benchmark_base::set_stopping_criterion(std::string criterion)
{
  m_stopping_criterion = std::move(criterion);
  m_criterion_params   = criterion_manager::get().get_criterion(m_stopping_criterion).get_params();
  return *this;
}

} // namespace nvbench
