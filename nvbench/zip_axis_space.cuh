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

#include <nvbench/axis_iteration_space.cuh>

namespace nvbench
{

struct zip_axis_space final : iteration_space_base
{
  zip_axis_space(std::vector<std::size_t> input_indices,
      std::vector<std::size_t> output_indices);
  ~zip_axis_space();

  std::unique_ptr<iteration_space_base> do_clone() const override;
  detail::axis_space_iterator do_get_iterator(axes_info info) const override;
  std::size_t do_get_size(const axes_info &info) const override;
  std::size_t do_get_active_count(const axes_info &info) const override;
};

} // namespace nvbench
