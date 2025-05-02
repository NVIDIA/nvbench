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
#include <nvbench/detail/state_generator.cuh>
#include <nvbench/detail/throw.cuh>
#include <nvbench/detail/transform_reduce.cuh>
#include <nvbench/device_info.cuh>
#include <nvbench/named_values.cuh>
#include <nvbench/type_axis.cuh>

#include <algorithm>
#include <cassert>
#include <exception>
#include <functional>
#include <numeric>

namespace nvbench::detail
{
// state_iterator ==============================================================

void state_iterator::add_iteration_space(const nvbench::detail::axis_space_iterator &iter)
{
  m_axes_count += iter.get_axis_value_indices().size();
  m_max_iteration *= iter.get_iteration_size();

  m_axis_space_iterators.push_back(std::move(iter));
}

[[nodiscard]] std::size_t state_iterator::get_number_of_states() const
{
  return this->m_max_iteration;
}

void state_iterator::init() { m_current_iteration = 0; }

[[nodiscard]] std::vector<axis_value_index> state_iterator::get_current_axis_value_indices() const
{
  std::vector<axis_value_index> info;
  info.reserve(m_axes_count);
  for (auto &iter : m_axis_space_iterators)
  {
    iter.update_axis_value_indices(info);
  }

  if (info.size() != m_axes_count)
  {
    NVBENCH_THROW(std::runtime_error,
                  "Internal error: State iterator has {} axes, but only {} were updated.",
                  m_axes_count,
                  info.size());
  }

  return info;
}

[[nodiscard]] bool state_iterator::iter_valid() const
{
  return m_current_iteration < m_max_iteration;
}

void state_iterator::next()
{
  m_current_iteration++;

  for (auto &iter : this->m_axis_space_iterators)
  {
    const auto rolled_over = iter.next();
    if (rolled_over)
    {
      continue;
    }
    break;
  }
}

// state_generator =============================================================

state_generator::state_generator(const benchmark_base &bench)
    : m_benchmark(bench)
{}

void state_generator::build_axis_configs()
{
  const axes_metadata &axes = m_benchmark.get_axes();
  const auto &type_spaces   = axes.get_type_iteration_spaces();
  const auto &value_spaces  = axes.get_value_iteration_spaces();

  state_iterator ti;
  state_iterator vi;

  // Reverse add type axes by index. This way the state_generator's cartesian
  // product of the type axes values will be enumerated in the same order as
  // nvbench::tl::cartesian_product<type_axes>. This is necessary to ensure
  // that the correct states are passed to the corresponding benchmark
  // instantiations.
  {
    const auto &axes_vec = axes.get_axes();
    std::for_each(type_spaces.crbegin(), //
                  type_spaces.crend(),
                  [&ti, &axes_vec](const auto &space) {
                    ti.add_iteration_space(space->get_iterator(axes_vec));
                  });
    std::for_each(value_spaces.begin(), //
                  value_spaces.end(),
                  [&vi, &axes_vec](const auto &space) {
                    vi.add_iteration_space(space->get_iterator(axes_vec));
                  });
  }

  m_type_axis_configs.clear();
  m_type_axis_configs.reserve(ti.get_number_of_states());

  m_non_type_axis_configs.clear();
  m_non_type_axis_configs.reserve(vi.get_number_of_states());

  for (ti.init(); ti.iter_valid(); ti.next())
  {
    auto &[config, active_mask] =
      m_type_axis_configs.emplace_back(std::make_pair(nvbench::named_values{}, true));

    for (const auto &info : ti.get_current_axis_value_indices())
    {
      const auto &axis = axes.get_type_axis(info.axis_name);

      active_mask &= axis.get_is_active(info.value_index);

      config.set_string(axis.get_name(), axis.get_input_string(info.value_index));
    }
  }

  for (vi.init(); vi.iter_valid(); vi.next())
  {
    auto &config = m_non_type_axis_configs.emplace_back();

    // Add non-type parameters to state:
    for (const auto &axis_value : vi.get_current_axis_value_indices())
    {
      switch (axis_value.axis_type)
      {
        default:
        case axis_type::type:
          assert("unreachable." && false);
          break;
        case axis_type::int64:
          config.set_int64(
            axis_value.axis_name,
            axes.get_int64_axis(axis_value.axis_name).get_value(axis_value.value_index));
          break;

        case axis_type::float64:
          config.set_float64(
            axis_value.axis_name,
            axes.get_float64_axis(axis_value.axis_name).get_value(axis_value.value_index));
          break;

        case axis_type::string:
          config.set_string(
            axis_value.axis_name,
            axes.get_string_axis(axis_value.axis_name).get_value(axis_value.value_index));
          break;
      } // switch (type)
    } // for (axis_values)
  }

  if (m_type_axis_configs.empty())
  {
    m_type_axis_configs.emplace_back(std::make_pair(nvbench::named_values{}, true));
  }
}

void state_generator::build_states()
{
  m_states.clear();

  const auto &devices = m_benchmark.get_devices();
  if (devices.empty())
  {
    this->add_states_for_device(std::nullopt);
  }
  else
  {
    for (const auto &device : devices)
    {
      this->add_states_for_device(device);
    }
  }
}

void state_generator::add_states_for_device(const std::optional<device_info> &device)
{
  const auto num_type_configs = m_type_axis_configs.size();
  for (std::size_t type_config_index = 0; type_config_index < num_type_configs; ++type_config_index)
  {
    const auto &[type_config, axis_mask] = m_type_axis_configs[type_config_index];
    if (!axis_mask)
    { // Don't generate inner vector if the type config is masked out.
      continue;
    }

    for (const auto &non_type_config : m_non_type_axis_configs)
    {
      // Concatenate the type + non_type configurations:
      nvbench::named_values config = type_config;
      config.append(non_type_config);

      // Create benchmark:
      m_states.push_back(nvbench::state{m_benchmark, std::move(config), device, type_config_index});
    }
  }
}

std::vector<nvbench::state> state_generator::create(const benchmark_base &bench)
{
  state_generator sg{bench};
  sg.build_axis_configs();
  sg.build_states();
  return std::move(sg.m_states);
}

} // namespace nvbench::detail
