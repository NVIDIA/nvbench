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

#include "linear_axis_space.cuh"

#include <nvbench/type_axis.cuh>

namespace nvbench
{

linear_axis_space::linear_axis_space(std::size_t in_index)
    : iteration_space_base({in_index})
{}

linear_axis_space::~linear_axis_space() = default;

detail::axis_space_iterator linear_axis_space::do_get_iterator(axes_info info) const
{
  auto update_func = [=](std::size_t inc_index, axes_info::iterator start, axes_info::iterator) {
    start->index = inc_index;
  };

  return detail::axis_space_iterator(info, info[0].size, update_func);
}

std::size_t linear_axis_space::do_get_size(const axes_info &info) const { return info[0].size; }

std::size_t linear_axis_space::do_get_active_count(const axes_info &info) const
{
  return info[0].active_size;
}

std::unique_ptr<iteration_space_base> linear_axis_space::do_clone() const
{
  return std::make_unique<linear_axis_space>(*this);
}

} // namespace nvbench
