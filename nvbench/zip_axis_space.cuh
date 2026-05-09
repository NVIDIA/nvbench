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
 * Provides linear forward iteration over multiple axes in lockstep
 *
 * Consider two axes with the following values:
 * { 0, 1, 2, 3, 4, 5 }
 * { 0, 1, 2, 3, 4, 5 }
 *
 * Using a zip_axis_space over these two axes will generate 6 values
 * ( {0,0}, {1,1}, {2,2}, ... ) instead of the default 36 values
 * ( {0,0}, {0,1}, {0,2}, ... ).
 *
 */
struct zip_axis_space final : iteration_space_base
{
  zip_axis_space(std::vector<std::size_t> input_axis_indices);
  ~zip_axis_space();

  std::unique_ptr<iteration_space_base> do_clone() const override;
  detail::axis_space_iterator do_get_iterator(axis_value_indices info) const override;
  std::size_t do_get_size(const axis_value_indices &info) const override;
  std::size_t do_get_active_count(const axis_value_indices &info) const override;
};

} // namespace nvbench
