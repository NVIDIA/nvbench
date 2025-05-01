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
#include <nvbench/detail/transform_reduce.cuh>
#include <nvbench/device_info.cuh>
#include <nvbench/named_values.cuh>
#include <nvbench/type_axis.cuh>

#include <algorithm>
#include <cassert>
#include <functional>
#include <numeric>

namespace nvbench::detail
{
// state_iterator ==============================================================

void state_iterator::add_iteration_space(const nvbench::detail::axis_space_iterator &iter)
{
  m_axes_count += iter.m_info.size();
  m_max_iteration *= iter.m_iteration_size;

  m_space.push_back(std::move(iter));
}

[[nodiscard]] std::size_t state_iterator::get_number_of_states() const
{
  return this->m_max_iteration;
}

void state_iterator::init()
{
  m_current_space     = 0;
  m_current_iteration = 0;
}

[[nodiscard]] std::vector<axis_index> state_iterator::get_current_indices() const
{
  std::vector<axis_index> indices;
  indices.reserve(m_axes_count);
  for (auto &m : m_space)
  {
    m.update_indices(indices);
  }
  // verify length
  return indices;
}

[[nodiscard]] bool state_iterator::iter_valid() const
{
  return m_current_iteration < m_max_iteration;
}

void state_iterator::next()
{
  m_current_iteration++;

  for (auto &&space : this->m_space)
  {
    auto rolled_over = space.next();
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
  const auto &type_space    = axes.get_type_iteration_space();
  const auto &value_space   = axes.get_value_iteration_space();

  state_iterator ti;
  state_iterator vi;

  // Reverse add type axes by index. This way the state_generator's cartesian
  // product of the type axes values will be enumerated in the same order as
  // nvbench::tl::cartesian_product<type_axes>. This is necessary to ensure
  // that the correct states are passed to the corresponding benchmark
  // instantiations.
  {
    const auto &axes_vec = axes.get_axes();
    std::for_each(type_space.crbegin(), type_space.crend(), [&ti, &axes_vec](const auto &space) {
      ti.add_iteration_space(space->get_iterator(axes_vec));
    });
    std::for_each(value_space.begin(), value_space.end(), [&vi, &axes_vec](const auto &space) {
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

    for (const auto &axis_info : ti.get_current_indices())
    {
      const auto &axis = axes.get_type_axis(axis_info.name);

      active_mask &= axis.get_is_active(axis_info.index);

      config.set_string(axis.get_name(), axis.get_input_string(axis_info.index));
    }
  }

  for (vi.init(); vi.iter_valid(); vi.next())
  {
    auto &config = m_non_type_axis_configs.emplace_back();

    // Add non-type parameters to state:
    for (const auto &axis_info : vi.get_current_indices())
    {
      switch (axis_info.type)
      {
        default:
        case axis_type::type:
          assert("unreachable." && false);
          break;
        case axis_type::int64:
          config.set_int64(axis_info.name,
                           axes.get_int64_axis(axis_info.name).get_value(axis_info.index));
          break;

        case axis_type::float64:
          config.set_float64(axis_info.name,
                             axes.get_float64_axis(axis_info.name).get_value(axis_info.index));
          break;

        case axis_type::string:
          config.set_string(axis_info.name,
                            axes.get_string_axis(axis_info.name).get_value(axis_info.index));
          break;
      } // switch (type)
    } // for (axis_info : current_indices)
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
