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
#include <cmath>
#include <limits>
#include <vector>

#include "test_asserts.cuh"

constexpr nvbench::int64_t max_invalid_measurements_cap = 1024;

nvbench::int64_t count_invalid_measurements_until_finished()
{
  nvbench::detail::stdrel_criterion criterion;
  criterion.initialize(nvbench::criterion_params{});
  // freshly initialized criterion starts as not is_finished
  ASSERT(!criterion.is_finished());

  const auto invalid_measurement              = nvbench::float64_t{0};
  nvbench::int64_t total_invalid_measurements = 0;
  while (!criterion.is_finished() && total_invalid_measurements < max_invalid_measurements_cap)
  {
    criterion.add_measurement(invalid_measurement);
    ++total_invalid_measurements;
  }
  ASSERT(criterion.is_finished());
  return total_invalid_measurements;
}

void test_const()
{
  nvbench::criterion_params params;
  nvbench::detail::stdrel_criterion criterion;
  using nvbench::detail::statistics::min_samples_for_noise_estimate;

  criterion.initialize(params);
  for (nvbench::int64_t i = 0; i < min_samples_for_noise_estimate; ++i)
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

  nvbench::detail::stdrel_criterion criterion;
  criterion.initialize(params);

  using nvbench::detail::statistics::min_samples_for_noise_estimate;

  std::vector<nvbench::float64_t> low_noise(min_samples_for_noise_estimate, 100.0);
  low_noise.back() = 101.0;
  for (nvbench::float64_t measurement : low_noise)
  {
    criterion.add_measurement(measurement);
  }
  ASSERT(criterion.is_finished());

  params.set_float64("max-noise", max_noise);
  criterion.initialize(params);

  std::vector<nvbench::float64_t> high_noise;
  high_noise.reserve(min_samples_for_noise_estimate);
  for (nvbench::int64_t i = 0; i < min_samples_for_noise_estimate; ++i)
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
  nvbench::detail::stdrel_criterion criterion;
  criterion.initialize(nvbench::criterion_params{});

  using nvbench::detail::statistics::min_samples_for_noise_estimate;
  for (nvbench::int64_t i = 1; i < min_samples_for_noise_estimate; ++i)
  {
    criterion.add_measurement(42.0);
  }
  ASSERT(!criterion.is_finished());
}

void test_stdrel_uses_sample_standard_deviation()
{
  using nvbench::detail::statistics::min_samples_for_noise_estimate;
  const nvbench::int64_t n   = std::max(nvbench::int64_t{26}, min_samples_for_noise_estimate);
  const nvbench::float64_t a = 6;
  const nvbench::float64_t b = 0;
  // for sequence t = a * i + b, 1 <= i <= n
  // mean = a*(n+1)/2 + b
  // variance = a^2/12 * (n^2 - 1)
  // for a, b, n = 6, 0, 26, mean = 81,
  // biased standard deviation = 45        (noise 0.5556)
  // unbiased standard deviation = 45.8912 (noise 0.5666)

  const nvbench::float64_t biased_noise = std::sqrt(static_cast<nvbench::float64_t>(n - 1) /
                                                    static_cast<nvbench::float64_t>(3 * (n + 1)));
  const nvbench::float64_t unbiased_noise =
    std::sqrt(static_cast<nvbench::float64_t>(n) / static_cast<nvbench::float64_t>(3 * (n + 1)));

  nvbench::criterion_params params;
  params.set_float64("max-noise", 0.5 * (biased_noise + unbiased_noise));

  nvbench::detail::stdrel_criterion criterion;
  criterion.initialize(params);

  for (int i = 1; i <= n; ++i)
  {
    const nvbench::float64_t measurement = a * static_cast<nvbench::float64_t>(i) + b;
    criterion.add_measurement(measurement);
  }

  ASSERT(!criterion.is_finished());
}

void test_stdrel_finishes_with_persistently_invalid_noise()
{
  const auto count = count_invalid_measurements_until_finished();
  ASSERT(count > 1);
}

void test_stdrel_invalid_noise_context_requires_min_time()
{
  nvbench::detail::stdrel_criterion criterion;
  criterion.initialize(nvbench::criterion_params{});

  const auto invalid_measurement              = nvbench::float64_t{0};
  nvbench::int64_t total_invalid_measurements = 0;
  while (!criterion.is_finished() && total_invalid_measurements < max_invalid_measurements_cap)
  {
    criterion.add_measurement(invalid_measurement);
    ++total_invalid_measurements;
  }
  ASSERT(criterion.is_finished());

  const auto min_time = nvbench::float64_t{1};
  {
    const auto total_samples = total_invalid_measurements;
    const auto total_time    = min_time / nvbench::float64_t{2};
    const auto min_samples   = total_invalid_measurements;
    const auto context =
      nvbench::stopping_context{total_samples, total_time, min_samples, min_time};
    ASSERT(!criterion.is_finished(context));
  }
  {
    const auto total_samples = total_invalid_measurements;
    const auto total_time    = min_time;
    const auto min_samples   = total_invalid_measurements;
    const auto context =
      nvbench::stopping_context{total_samples, total_time, min_samples, min_time};
    ASSERT(criterion.is_finished(context));
  }
}

void test_stdrel_context_requires_min_samples_and_min_time()
{
  nvbench::detail::stdrel_criterion criterion;
  criterion.initialize(nvbench::criterion_params{});

  using nvbench::detail::statistics::min_samples_for_noise_estimate;
  for (nvbench::int64_t i = 0; i < min_samples_for_noise_estimate; ++i)
  {
    criterion.add_measurement(42.0);
  }

  ASSERT(criterion.is_finished());
  {
    const auto total_samples = min_samples_for_noise_estimate - 1;
    const auto total_time    = nvbench::float64_t{42};
    const auto min_samples   = min_samples_for_noise_estimate;
    const auto min_time      = nvbench::float64_t{0};
    const auto context =
      nvbench::stopping_context{total_samples, total_time, min_samples, min_time};
    ASSERT(!criterion.is_finished(context));
  }
  {
    const auto total_samples = min_samples_for_noise_estimate;
    const auto total_time    = nvbench::float64_t{0.1};
    const auto min_samples   = min_samples_for_noise_estimate;
    const auto min_time      = nvbench::float64_t{1};
    const auto context =
      nvbench::stopping_context{total_samples, total_time, min_samples, min_time};
    ASSERT(!criterion.is_finished(context));
  }
  {
    const auto total_samples = min_samples_for_noise_estimate;
    const auto total_time    = nvbench::float64_t{1};
    const auto min_samples   = min_samples_for_noise_estimate;
    const auto min_time      = nvbench::float64_t{1};
    const auto context =
      nvbench::stopping_context{total_samples, total_time, min_samples, min_time};
    ASSERT(criterion.is_finished(context));
  }
}

int main()
{
  test_const();
  test_stdrel();
  test_stdrel_needs_enough_samples();
  test_stdrel_uses_sample_standard_deviation();
  test_stdrel_finishes_with_persistently_invalid_noise();
  test_stdrel_invalid_noise_context_requires_min_time();
  test_stdrel_context_requires_min_samples_and_min_time();
}
