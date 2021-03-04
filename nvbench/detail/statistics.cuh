/*
 *  Copyright 2020 NVIDIA Corporation
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

#include <nvbench/detail/transform_reduce.cuh>

#include <functional>
#include <limits>
#include <numeric>
#include <vector>

namespace nvbench::detail
{

/**
 * Given a vector of samples and the precomputed sum of all samples in the
 * vector, return a measure of the noise in the samples.
 *
 * The noise metric is the relative unbiased sample standard deviation
 * expressed as a percentage: (std_dev / mean) * 100.
 */
inline nvbench::float64_t
compute_noise(const std::vector<nvbench::float64_t> &data,
              nvbench::float64_t sum)
{
  const auto num = static_cast<nvbench::float64_t>(data.size());
  if (num < 5) // don't bother with low sample sizes.
  {
    return std::numeric_limits<nvbench::float64_t>::infinity();
  }

  const auto mean = sum / num;
  const auto variance =
    nvbench::detail::transform_reduce(data.cbegin(),
                                      data.cend(),
                                      0.,
                                      std::plus<>{},
                                      [mean](nvbench::float64_t val) {
                                        val -= mean;
                                        val *= val;
                                        return val;
                                      }) /
    (num - 1);
  const auto abs_stdev = std::sqrt(variance);
  const auto rel_stdev = abs_stdev / mean;
  return rel_stdev * 100.;
}

} // namespace nvbench::detail
