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

#include "axis_iteration_space.cuh"

#include <nvbench/type_axis.cuh>

namespace nvbench
{

axis_space_base::axis_space_base(std::vector<std::size_t> input_indices,
                                 std::vector<std::size_t> output_indices)
    : m_input_indices(std::move(input_indices))
    , m_output_indices(std::move(output_indices))
{}

axis_space_base::~axis_space_base() = default;

std::unique_ptr<axis_space_base> axis_space_base::clone() const
{
  auto clone = this->do_clone();
  return clone;
}

std::vector<std::unique_ptr<axis_space_base>>
axis_space_base::clone_as_linear() const
{
  std::vector<std::unique_ptr<axis_space_base>> clones;
  clones.reserve(m_input_indices.size());

  for (std::size_t i = 0; i < m_input_indices.size(); ++i)
  {
    clones.push_back(
      std::make_unique<nvbench::linear_axis_space>(m_input_indices[i],
                                                   m_output_indices[i]));
  }

  return clones;
}

namespace
{
nvbench::axis_space_base::axes_info
get_axes_info(const nvbench::axis_space_base::axes_type &axes,
              const std::vector<std::size_t> &indices)
{
  nvbench::axis_space_base::axes_info info;
  info.reserve(indices.size());
  for (auto &n : indices)
  {
    info.emplace_back(axes[n].get());
  }
  return info;
}
} // namespace

detail::axis_space_iterator axis_space_base::iter(const axes_type &axes) const
{

  return this->do_iter(get_axes_info(axes, m_input_indices));
}

std::size_t axis_space_base::size(const axes_type &axes) const
{
  return this->do_size(get_axes_info(axes, m_input_indices));
}
std::size_t axis_space_base::valid_count(const axes_type &axes) const
{
  return this->do_valid_count(get_axes_info(axes, m_input_indices));
}

bool axis_space_base::contains(std::size_t in_index) const
{
  auto iter =
    std::find_if(m_input_indices.cbegin(),
                 m_input_indices.cend(),
                 [&in_index](const auto &i) { return i == in_index; });
  return iter != m_input_indices.end();
}

linear_axis_space::linear_axis_space(std::size_t in_index,
                                     std::size_t out_index)
    : axis_space_base({std::move(in_index)}, {out_index})
{}

linear_axis_space::~linear_axis_space() = default;

detail::axis_space_iterator linear_axis_space::do_iter(axes_info info) const
{
  std::size_t loc(m_output_indices[0]);
  auto update_func = [=](std::size_t inc_index,
                         std::vector<detail::axis_index> &indices) {
    indices[loc]       = info[0];
    indices[loc].index = inc_index;
  };

  return detail::make_space_iterator(1, info[0].size, update_func);
}

std::size_t linear_axis_space::do_size(const axes_info &info) const
{
  return info[0].size;
}

std::size_t linear_axis_space::do_valid_count(const axes_info &info) const
{
  return info[0].active_size;
}

std::unique_ptr<axis_space_base> linear_axis_space::do_clone() const
{
  return std::make_unique<linear_axis_space>(*this);
}

zip_axis_space::zip_axis_space(std::vector<std::size_t> input_indices,
                               std::vector<std::size_t> output_indices)
    : axis_space_base(std::move(input_indices), std::move(output_indices))
{}

zip_axis_space::~zip_axis_space() = default;

detail::axis_space_iterator zip_axis_space::do_iter(axes_info info) const
{
  std::vector<std::size_t> locs = m_output_indices;
  auto update_func              = [=](std::size_t inc_index,
                         std::vector<detail::axis_index> &indices) {
    for (std::size_t i = 0; i < info.size(); ++i)
    {
      detail::axis_index temp = info[i];
      temp.index              = inc_index;
      indices[locs[i]]        = temp;
    }
  };

  return detail::make_space_iterator(locs.size(), info[0].size, update_func);
}

std::size_t zip_axis_space::do_size(const axes_info &info) const
{
  return info[0].size;
}

std::size_t zip_axis_space::do_valid_count(const axes_info &info) const
{
  return info[0].active_size;
}

std::unique_ptr<axis_space_base> zip_axis_space::do_clone() const
{
  return std::make_unique<zip_axis_space>(*this);
}

user_axis_space::user_axis_space(std::vector<std::size_t> input_indices,
                                 std::vector<std::size_t> output_indices)
    : axis_space_base(std::move(input_indices), std::move(output_indices))
{}
user_axis_space::~user_axis_space() = default;

} // namespace nvbench
