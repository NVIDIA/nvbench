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
