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

#include "zip_axis_space.cuh"

#include <nvbench/type_axis.cuh>

namespace nvbench
{

zip_axis_space::zip_axis_space(std::vector<std::size_t> input_indices,
                               std::vector<std::size_t> output_indices)
    : iteration_space_base(std::move(input_indices), std::move(output_indices))
{}

zip_axis_space::~zip_axis_space() = default;

detail::axis_space_iterator zip_axis_space::do_get_iterator(axes_info info) const
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

  return detail::axis_space_iterator(locs.size(), info[0].size, update_func);
}

std::size_t zip_axis_space::do_get_size(const axes_info &info) const
{
  return info[0].size;
}

std::size_t zip_axis_space::do_get_active_count(const axes_info &info) const
{
  return info[0].active_size;
}

std::unique_ptr<iteration_space_base> zip_axis_space::do_clone() const
{
  return std::make_unique<zip_axis_space>(*this);
}

} // namespace nvbench
