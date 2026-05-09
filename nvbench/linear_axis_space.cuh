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
 * Provides linear forward iteration over a single axis.
 *
 * The default for all axes added to a benchmark
 */
struct linear_axis_space final : iteration_space_base
{
  linear_axis_space(std::size_t axis_index);
  ~linear_axis_space();

  std::unique_ptr<iteration_space_base> do_clone() const override;
  detail::axis_space_iterator do_get_iterator(axis_value_indices info) const override;
  std::size_t do_get_size(const axis_value_indices &info) const override;
  std::size_t do_get_active_count(const axis_value_indices &info) const override;
};

} // namespace nvbench
