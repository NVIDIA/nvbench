/*
 *  Copyright 2026 NVIDIA Corporation
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

#include <nvbench/detail/sample_count_criterion.cuh>
#include <nvbench/stopping_criterion.cuh>

#include "test_asserts.cuh"

void test_default_target_samples()
{
  nvbench::detail::sample_count_criterion criterion;
  criterion.initialize(nvbench::criterion_params{});

  for (int i = 0; i < 99; ++i)
  {
    criterion.add_measurement(1.0);
    ASSERT(!criterion.is_finished());
  }

  criterion.add_measurement(1.0);
  ASSERT(criterion.is_finished());
}

void test_custom_target_samples()
{
  nvbench::criterion_params params;
  params.set_int64("target-samples", 3);

  nvbench::detail::sample_count_criterion criterion;
  criterion.initialize(params);

  criterion.add_measurement(1.0);
  ASSERT(!criterion.is_finished());

  criterion.add_measurement(1.0);
  ASSERT(!criterion.is_finished());

  criterion.add_measurement(1.0);
  ASSERT(criterion.is_finished());
}

void test_target_samples_one()
{
  nvbench::criterion_params params;
  params.set_int64("target-samples", 1);

  nvbench::detail::sample_count_criterion criterion;
  criterion.initialize(params);

  ASSERT(!criterion.is_finished());

  criterion.add_measurement(1.0);
  ASSERT(criterion.is_finished());
}

void test_context_ignores_global_floors()
{
  nvbench::criterion_params params;
  params.set_int64("target-samples", 3);

  nvbench::detail::sample_count_criterion criterion;
  criterion.initialize(params);

  for (int i = 0; i < 3; ++i)
  {
    criterion.add_measurement(1.0);
  }

  ASSERT(criterion.is_finished());
  ASSERT(criterion.is_finished(nvbench::stopping_context{3, 0.0, 10, 1.0}));
}

void test_non_positive_target_samples()
{
  for (const auto target_samples : {0, -1})
  {
    nvbench::criterion_params params;
    params.set_int64("target-samples", target_samples);

    nvbench::detail::sample_count_criterion criterion;
    ASSERT_THROWS_ANY(criterion.initialize(params));
  }
}

int main()
{
  test_default_target_samples();
  test_custom_target_samples();
  test_target_samples_one();
  test_context_ignores_global_floors();
  test_non_positive_target_samples();
}
