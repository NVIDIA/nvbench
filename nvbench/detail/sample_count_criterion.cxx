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

namespace nvbench::detail
{

sample_count_criterion::sample_count_criterion()
    : stopping_criterion_base{"sample-count", {{"target-samples", nvbench::int64_t{100}}}}
{}

void sample_count_criterion::do_initialize() { m_total_samples = 0; }

void sample_count_criterion::do_add_measurement(nvbench::float64_t) { ++m_total_samples; }

bool sample_count_criterion::do_is_finished()
{
  return m_total_samples >= m_params.get_int64("target-samples");
}

} // namespace nvbench::detail
