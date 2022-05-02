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

#pragma once

#include <nvbench/iteration_space_base.cuh>

namespace nvbench
{

/*!
 * Provides user defined iteration over multiple axes
 *
 * Consider two axi with the following values:
 * { 0, 1, 2, 3, 4, 5 }
 * { 0, 1, 2, 3, 4, 5 }
 *
 * If we wanted to provide an axis space that skipped every third value
 * We would implement it like this:
 *
 * struct every_third final : nvbench::user_axis_space
 * {
 *   every_third(std::vector<std::size_t> input_indices,
 *               std::vector<std::size_t> output_indices)
 *       : nvbench::user_axis_space(std::move(input_indices),
 *                                  std::move(output_indices))
 *   {}
 *
 *   nvbench::detail::axis_space_iterator do_get_iterator(axes_info info) const
 *   {
 *     // our increment function
 *     auto adv_func = [&, info](std::size_t &inc_index, std::size_t len) -> bool {
 *       inc_index += 3;
 *       return inc_index >= len;
 *     };
 *
 *     // our update function
 *     std::vector<std::size_t> locs = m_output_indices;
 *     auto update_func              = [=](std::size_t inc_index,
 *                            std::vector<detail::axis_index> &indices) {
 *       for (std::size_t i = 0; i < info.size(); ++i)
 *       {
 *         detail::axis_index temp = info[i];
 *         temp.index              = inc_index;
 *         indices[locs[i]]        = temp;
 *       }
 *     };
 *    return detail::make_space_iterator(locs.size(), (info[0].size/3), adv_func, update_func);
 *   }
 *
 *   std::size_t do_get_size(const axes_info &info) const { return (info[0].size/3); }
 *   ...
 *
 */
struct user_axis_space : iteration_space_base
{
  user_axis_space(std::vector<std::size_t> input_indices,
                  std::vector<std::size_t> output_indices);
  ~user_axis_space();
};

using make_user_space_signature =
  std::unique_ptr<iteration_space_base>(std::vector<std::size_t> input_indices,
                                   std::vector<std::size_t> output_indices);

} // namespace nvbench
