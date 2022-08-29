/*
 *  Copyright 2022 NVIDIA Corporation
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

#include "iteration_space_base.cuh"

#include <nvbench/type_axis.cuh>

namespace nvbench
{

iteration_space_base::iteration_space_base(std::vector<std::size_t> input_indices,
                                 std::vector<std::size_t> output_indices)
    : m_input_indices(std::move(input_indices))
    , m_output_indices(std::move(output_indices))
{}

iteration_space_base::~iteration_space_base() = default;

std::unique_ptr<iteration_space_base> iteration_space_base::clone() const
{
  auto clone = this->do_clone();
  return clone;
}

namespace
{
nvbench::iteration_space_base::axes_info
get_axes_info(const nvbench::iteration_space_base::axes_type &axes,
              const std::vector<std::size_t> &indices)
{
  nvbench::iteration_space_base::axes_info info;
  info.reserve(indices.size());
  for (auto &n : indices)
  {
    info.emplace_back(axes[n].get());
  }
  return info;
}
} // namespace

detail::axis_space_iterator iteration_space_base::get_iterator(const axes_type &axes) const
{

  return this->do_get_iterator(get_axes_info(axes, m_input_indices));
}

std::size_t iteration_space_base::get_size(const axes_type &axes) const
{
  return this->do_get_size(get_axes_info(axes, m_input_indices));
}
std::size_t iteration_space_base::get_active_count(const axes_type &axes) const
{
  return this->do_get_active_count(get_axes_info(axes, m_input_indices));
}

} // namespace nvbench
