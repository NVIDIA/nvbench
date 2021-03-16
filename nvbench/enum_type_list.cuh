/*
 *  Copyright 2021 NVIDIA Corporation
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

#include <nvbench/type_list.cuh>
#include <nvbench/type_strings.cuh>

#include <type_traits>

namespace nvbench
{

/*!
 * \brief Helper utility that generates a `type_list` of
 * `std::integral_constant`s.
 *
 * \relatesalso NVBENCH_DECLARE_ENUM_TYPE_STRINGS
 */
template <typename T, T... Ts>
using enum_type_list = nvbench::type_list<std::integral_constant<T, Ts>...>;
} // namespace nvbench

/*!
 * \brief Declare `type_string`s for an `enum_type_list`.
 *
 * Given an enum type `T` and two callables that produce input and description
 * strings, declare a specialization for
 * `nvbench::type_string<std::integral_constant<T, Value>>`.
 *
 * Must be used from global namespace scope.
 *
 * \relatesalso nvbench::enum_type_list
 */
#define NVBENCH_DECLARE_ENUM_TYPE_STRINGS(T,                                   \
                                          input_generator,                     \
                                          description_generator)               \
  namespace nvbench                                                            \
  {                                                                            \
  template <T Value>                                                           \
  struct type_strings<std::integral_constant<T, Value>>                        \
  {                                                                            \
    static std::string input_string() { return input_generator(Value); }       \
    static std::string description() { return description_generator(Value); }  \
  };                                                                           \
  }
