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

#include <nvbench/detail/throw.cuh>
#include <nvbench/type_axis.cuh>

#include <exception>

namespace nvbench
{

zip_axis_space::zip_axis_space(std::vector<std::size_t> input_axis_indices)
    : iteration_space_base(std::move(input_axis_indices))
{}

zip_axis_space::~zip_axis_space() = default;

detail::axis_space_iterator zip_axis_space::do_get_iterator(axis_value_indices info) const
{
  const auto axis_size = info[0].axis_size;
  for (const auto &axis : info)
  {
    if (axis.axis_active_size != axis_size)
    {
      NVBENCH_THROW(std::runtime_error, "%s", "All zipped axes must have the same size.");
    }
  }

  auto update_func = [](std::size_t current_iteration,
                        axis_value_indices::iterator start_axis_value_info,
                        axis_value_indices::iterator end_axis_value_info) {
    for (; start_axis_value_info != end_axis_value_info; ++start_axis_value_info)
    {
      start_axis_value_info->value_index = current_iteration;
    }
  };

  return detail::axis_space_iterator(std::move(info), axis_size, update_func);
}

std::size_t zip_axis_space::do_get_size(const axis_value_indices &info) const
{
  return info[0].axis_size;
}

std::size_t zip_axis_space::do_get_active_count(const axis_value_indices &info) const
{
  return info[0].axis_active_size;
}

std::unique_ptr<iteration_space_base> zip_axis_space::do_clone() const
{
  return std::make_unique<zip_axis_space>(*this);
}

} // namespace nvbench
