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

#include <type_traits>
#include <vector>

namespace nvbench
{

namespace detail
{
template <typename T>
using range_output_t =
  std::conditional_t<std::is_floating_point_v<T>, nvbench::float64_t, nvbench::int64_t>;
}

template <typename InT, typename OutT = nvbench::detail::range_output_t<InT>>
auto range(InT start, InT end, InT stride = InT{1})
{
  if constexpr (std::is_floating_point_v<InT>)
  { // Pad end to account for floating point errors:
    end += (stride / InT{2});
  }
  using result_t = std::vector<OutT>;
  result_t result;
  for (; start <= end; start += stride)
  {
    result.push_back(static_cast<OutT>(start));
  }
  return result;
}

} // namespace nvbench
