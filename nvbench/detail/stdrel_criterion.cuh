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

#include <nvbench/config.cuh>

#if defined(NVBENCH_IMPLICIT_SYSTEM_HEADER_GCC)
#pragma GCC system_header
#elif defined(NVBENCH_IMPLICIT_SYSTEM_HEADER_CLANG)
#pragma clang system_header
#elif defined(NVBENCH_IMPLICIT_SYSTEM_HEADER_MSVC)
#pragma system_header
#endif

#include <nvbench/detail/ring_buffer.cuh>
#include <nvbench/detail/statistics.cuh>
#include <nvbench/stopping_criterion.cuh>
#include <nvbench/types.cuh>

namespace nvbench::detail
{

class stdrel_criterion final : public stopping_criterion_base
{
  // state
  nvbench::int64_t m_consecutive_invalid_noise_estimates{};
  nvbench::detail::statistics::online_mean_variance m_cuda_times_summary{};
  nvbench::detail::ring_buffer<nvbench::float64_t> m_noise_tracker{512};

public:
  stdrel_criterion();

protected:
  virtual void do_initialize() override;
  virtual void do_add_measurement(nvbench::float64_t measurement) override;
  virtual bool do_is_finished() override;
};

} // namespace nvbench::detail
