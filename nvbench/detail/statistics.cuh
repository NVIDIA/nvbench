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

#include <nvbench/types.cuh>

#include <nvbench/detail/transform_reduce.cuh>

#include <cmath>
#include <functional>
#include <iterator>
#include <limits>
#include <type_traits>

namespace nvbench::detail::statistics
{

/**
 * Computes and returns the unbiased sample standard deviation.
 *
 * If the input has fewer than 5 sample, infinity is returned.
 */
template <typename Iter, typename ValueType = typename std::iterator_traits<Iter>::value_type>
ValueType standard_deviation(Iter first, Iter last, ValueType mean)
{
  static_assert(std::is_floating_point_v<ValueType>);

  const auto num = last - first;
  if (num < 5) // don't bother with low sample sizes.
  {
    return std::numeric_limits<ValueType>::infinity();
  }

  const auto variance = nvbench::detail::transform_reduce(first,
                                                          last,
                                                          ValueType{},
                                                          std::plus<>{},
                                                          [mean](auto val) {
                                                            val -= mean;
                                                            val *= val;
                                                            return val;
                                                          }) /
                        static_cast<ValueType>((num - 1));
  return std::sqrt(variance);
}

} // namespace nvbench::detail::statistics
