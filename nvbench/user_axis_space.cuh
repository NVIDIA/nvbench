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
 * Provides user defined iteration over one or more axes
 *
 * If we wanted to provide an axis space that only returns every third
 * value in an axis we would implement it like this:
 *
 * struct every_third final : nvbench::user_axis_space
 * {
 *   explicit every_third(std::vector<std::size_t> input_indices)
 *       : nvbench::user_axis_space(std::move(input_indices))
 *   {}
 *
 *   nvbench::detail::axis_space_iterator do_get_iterator(axes_info info) const
 *   {
 *     // our increment function
 *     auto adv_func = [](std::size_t &inc_index,
 *                        std::size_t len) -> bool {
 *       inc_index += 3;
 *       return inc_index >= len;
 *     };
 *
 *     // our update function
 *     auto update_func = [](std::size_t inc_index,
 *                           axes_info::iterator start,
 *                           axes_info::iterator end) {
 *       for (; start != end; ++start) {
 *         start->index = inc_index;
 *       }
 *     };
 *    return detail::axis_space_iterator(info, (info[0].size/3),
 *                                       adv_func, update_func);
 *   }
 *
 *   std::size_t do_get_size(const axes_info &info) const
 *   {
 *     return (info[0].size/3);
 *   }
 *   ...
 * };
 */
struct user_axis_space : iteration_space_base
{
  user_axis_space(std::vector<std::size_t> input_indices);
  ~user_axis_space();
};

using make_user_space_signature =
  std::unique_ptr<iteration_space_base>(std::vector<std::size_t> input_indices);

} // namespace nvbench
