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

#pragma once

#include <nvbench/config.cuh>

#if defined(NVBENCH_IMPLICIT_SYSTEM_HEADER_GCC)
#pragma GCC system_header
#elif defined(NVBENCH_IMPLICIT_SYSTEM_HEADER_CLANG)
#pragma clang system_header
#elif defined(NVBENCH_IMPLICIT_SYSTEM_HEADER_MSVC)
#pragma system_header
#endif

#include <nvbench/detail/statistics.cuh>
#include <nvbench/printer_base.cuh>
#include <nvbench/stopping_criterion.cuh>
#include <nvbench/types.cuh>

#include <fmt/format.h>

#include <optional>
#include <string>

namespace nvbench::detail
{

inline std::optional<nvbench::float64_t>
get_float64_criterion_param(const nvbench::criterion_params &params, const std::string &name)
{
  if (!params.has_value(name))
  {
    return std::nullopt;
  }
  return params.get_float64(name);
}

inline void log_measurement_timeout_warnings(nvbench::printer_base &printer,
                                             const nvbench::criterion_params &criterion_params,
                                             nvbench::float64_t timeout,
                                             nvbench::int64_t total_samples,
                                             nvbench::int64_t min_samples,
                                             nvbench::float64_t accumulated_time,
                                             std::optional<nvbench::float64_t> stdev_noise)
{
  const auto max_noise = get_float64_criterion_param(criterion_params, "max-noise");
  const auto min_time  = get_float64_criterion_param(criterion_params, "min-time");

  const auto enough_samples_for_noise =
    statistics::has_enough_samples_for_noise_estimate(total_samples);
  if (max_noise && !enough_samples_for_noise)
  {
    printer.log(nvbench::log_level::warn,
                fmt::format("Current measurement timed out ({:0.2f}s) "
                            "before accumulating enough samples to estimate noise ({} < {})",
                            timeout,
                            total_samples,
                            statistics::min_samples_for_noise_estimate));
  }
  else if (max_noise && !stdev_noise)
  {
    printer.log(nvbench::log_level::warn,
                fmt::format("Current measurement timed out ({:0.2f}s) "
                            "while unable to estimate noise for max-noise",
                            timeout));
  }
  else if (max_noise && stdev_noise && *stdev_noise > *max_noise)
  {
    printer.log(nvbench::log_level::warn,
                fmt::format("Current measurement timed out ({:0.2f}s) "
                            "while over noise threshold ({:0.2f}% > "
                            "{:0.2f}%)",
                            timeout,
                            *stdev_noise * 100,
                            *max_noise * 100));
  }
  if (total_samples < min_samples)
  {
    printer.log(nvbench::log_level::warn,
                fmt::format("Current measurement timed out ({:0.2f}s) "
                            "before accumulating min_samples ({} < {})",
                            timeout,
                            total_samples,
                            min_samples));
  }
  if (min_time && accumulated_time < *min_time)
  {
    printer.log(nvbench::log_level::warn,
                fmt::format("Current measurement timed out ({:0.2f}s) "
                            "before accumulating min_time ({:0.2f}s < "
                            "{:0.2f}s)",
                            timeout,
                            accumulated_time,
                            *min_time));
  }
}

} // namespace nvbench::detail
