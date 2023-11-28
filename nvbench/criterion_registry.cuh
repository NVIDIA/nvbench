/*
 *  Copyright 2023 NVIDIA Corporation
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

#include <nvbench/types.cuh>
#include <nvbench/stopping_criterion.cuh>
#include <nvbench/detail/stdrel_criterion.cuh>
#include <nvbench/detail/entropy_criterion.cuh>

#include <unordered_map>
#include <memory>

namespace nvbench
{

class criterion_registry
{
  std::unordered_map<std::string, std::unique_ptr<nvbench::stopping_criterion>> m_map;

  criterion_registry();

public:
  static criterion_registry &instance();

  static nvbench::stopping_criterion* get(const std::string& name);

  static bool register_criterion(std::string name,
                                 std::unique_ptr<nvbench::stopping_criterion> criterion);

  static nvbench::stopping_criterion::params_description get_params_description();
};

} // namespace nvbench
