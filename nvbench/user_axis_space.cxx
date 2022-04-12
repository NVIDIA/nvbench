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

#include "user_axis_space.cuh"

#include <nvbench/type_axis.cuh>

namespace nvbench
{

user_axis_space::user_axis_space(std::vector<std::size_t> input_indices,
                                 std::vector<std::size_t> output_indices)
    : iteration_space_base(std::move(input_indices), std::move(output_indices))
{}
user_axis_space::~user_axis_space() = default;

} // namespace nvbench
