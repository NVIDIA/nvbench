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

#include <nvbench/detail/stdrel_criterion.cuh>
#include <nvbench/stopping_criterion.cuh>
#include <nvbench/types.cuh>

#include <limits>
#include <vector>

#include "test_asserts.cuh"

void test_const()
{
  nvbench::criterion_params params;
  nvbench::detail::stdrel_criterion criterion;

  criterion.initialize(params);
  for (int i = 0; i < 5; i++)
  { // nvbench wants at least 5 to compute the standard deviation
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

  const std::vector<nvbench::float64_t> low_noise{100.0, 100.0, 100.0, 101.0, 101.0};
  for (nvbench::float64_t measurement : low_noise)
  {
    criterion.add_measurement(measurement);
  }
  ASSERT(criterion.is_finished());

  params.set_float64("max-noise", max_noise);
  criterion.initialize(params);

  const std::vector<nvbench::float64_t> high_noise{10.0, 20.0, 30.0, 40.0, 50.0};
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

  for (int i = 0; i < 4; ++i)
  {
    criterion.add_measurement(42.0);
  }
  ASSERT(!criterion.is_finished());
}

void test_stdrel_finishes_with_persistently_invalid_noise()
{
  nvbench::criterion_params params;
  params.set_float64("min-time", 0.0);

  nvbench::detail::stdrel_criterion criterion;
  criterion.initialize(params);

  // Force invalid relative-IQR estimates while still satisfying min-time.
  const auto invalid_measurement = std::numeric_limits<nvbench::float64_t>::infinity();
  for (int i = 0; i < 67; ++i)
  {
    criterion.add_measurement(invalid_measurement);
  }
  ASSERT(!criterion.is_finished());

  criterion.add_measurement(invalid_measurement);
  ASSERT(criterion.is_finished());
}

int main()
{
  test_const();
  test_stdrel();
  test_stdrel_needs_enough_samples();
  test_stdrel_finishes_with_persistently_invalid_noise();
}
