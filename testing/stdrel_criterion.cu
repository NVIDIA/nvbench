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

#include <nvbench/detail/statistics.cuh>
#include <nvbench/detail/stdrel_criterion.cuh>
#include <nvbench/stopping_criterion.cuh>
#include <nvbench/types.cuh>

#include <algorithm>
#include <limits>
#include <vector>

#include "test_asserts.cuh"

nvbench::int64_t count_invalid_measurements_until_finished()
{
  nvbench::criterion_params params;
  params.set_float64("min-time", 0.0);

  nvbench::detail::stdrel_criterion criterion;
  criterion.initialize(params);

  const auto invalid_measurement = std::numeric_limits<nvbench::float64_t>::infinity();
  constexpr nvbench::int64_t max_invalid_measurements = 1024;
  nvbench::int64_t total_invalid_measurements         = 0;
  while (!criterion.is_finished() && total_invalid_measurements < max_invalid_measurements)
  {
    criterion.add_measurement(invalid_measurement);
    ++total_invalid_measurements;
  }
  ASSERT(total_invalid_measurements > 0);
  ASSERT(criterion.is_finished());
  return total_invalid_measurements;
}

void test_const()
{
  nvbench::criterion_params params;
  nvbench::detail::stdrel_criterion criterion;

  criterion.initialize(params);
  for (nvbench::int64_t i = 0; i < nvbench::detail::statistics::min_samples_for_noise_estimate; ++i)
  {
    criterion.add_measurement(42.0);
  }
  ASSERT(criterion.is_finished());
}

void test_stdrel()
{
  const nvbench::float64_t max_noise = 0.1;

  nvbench::criterion_params params;
  params.set_float64("max-noise", max_noise);
  params.set_float64("min-time", 0.0);

  nvbench::detail::stdrel_criterion criterion;
  criterion.initialize(params);

  std::vector<nvbench::float64_t> low_noise(
    nvbench::detail::statistics::min_samples_for_noise_estimate,
    100.0);
  low_noise.back() = 101.0;
  for (nvbench::float64_t measurement : low_noise)
  {
    criterion.add_measurement(measurement);
  }
  ASSERT(criterion.is_finished());

  params.set_float64("max-noise", max_noise);
  criterion.initialize(params);

  std::vector<nvbench::float64_t> high_noise;
  high_noise.reserve(nvbench::detail::statistics::min_samples_for_noise_estimate);
  for (nvbench::int64_t i = 0; i < nvbench::detail::statistics::min_samples_for_noise_estimate; ++i)
  {
    high_noise.push_back(static_cast<nvbench::float64_t>(i + 1) * 10.0);
  }
  for (nvbench::float64_t measurement : high_noise)
  {
    criterion.add_measurement(measurement);
  }
  ASSERT(!criterion.is_finished());
}

void test_stdrel_needs_enough_samples()
{
  nvbench::criterion_params params;
  params.set_float64("min-time", 0.0);

  nvbench::detail::stdrel_criterion criterion;
  criterion.initialize(params);

  for (nvbench::int64_t i = 1; i < nvbench::detail::statistics::min_samples_for_noise_estimate; ++i)
  {
    criterion.add_measurement(42.0);
  }
  ASSERT(!criterion.is_finished());
}

void test_stdrel_finishes_with_persistently_invalid_noise()
{
  [[maybe_unused]] const auto count = count_invalid_measurements_until_finished();
}

void test_stdrel_invalid_noise_bypasses_min_time(nvbench::float64_t invalid_measurement)
{
  nvbench::criterion_params params;
  params.set_float64("min-time", 1.0);

  nvbench::detail::stdrel_criterion criterion;
  criterion.initialize(params);

  constexpr nvbench::int64_t max_invalid_measurements = 1024;
  nvbench::int64_t total_invalid_measurements         = 0;
  while (!criterion.is_finished() && total_invalid_measurements < max_invalid_measurements)
  {
    criterion.add_measurement(invalid_measurement);
    ++total_invalid_measurements;
  }
  ASSERT(total_invalid_measurements > 0);
  ASSERT(criterion.is_finished());
}

void test_stdrel_invalid_noise_count_resets_after_valid_noise()
{
  const auto invalid_measurement  = std::numeric_limits<nvbench::float64_t>::infinity();
  const auto invalid_finish_count = count_invalid_measurements_until_finished();
  const auto initial_invalid_measurements =
    std::max(nvbench::detail::statistics::min_samples_for_noise_estimate, invalid_finish_count / 4);
  const auto valid_measurements = invalid_finish_count - initial_invalid_measurements;
  ASSERT(valid_measurements > 0);

  nvbench::criterion_params params;
  params.set_float64("max-noise", -1.0);
  params.set_float64("min-time", 0.0);

  nvbench::detail::stdrel_criterion criterion;
  criterion.initialize(params);

  for (nvbench::int64_t i = 0; i < initial_invalid_measurements; ++i)
  {
    criterion.add_measurement(invalid_measurement);
    ASSERT(!criterion.is_finished());
  }

  for (nvbench::int64_t i = 0; i < valid_measurements; ++i)
  {
    criterion.add_measurement(100.0);
    ASSERT(!criterion.is_finished());
  }

  criterion.add_measurement(invalid_measurement);
  ASSERT(!criterion.is_finished());
}

int main()
{
  test_const();
  test_stdrel();
  test_stdrel_needs_enough_samples();
  test_stdrel_finishes_with_persistently_invalid_noise();
  test_stdrel_invalid_noise_bypasses_min_time(nvbench::float64_t{});
  test_stdrel_invalid_noise_bypasses_min_time(std::numeric_limits<nvbench::float64_t>::infinity());
  test_stdrel_invalid_noise_count_resets_after_valid_noise();
}
