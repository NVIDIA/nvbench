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

#include <nvbench/detail/measure_timeout_warnings.cuh>
#include <nvbench/detail/statistics.cuh>
#include <nvbench/printer_base.cuh>
#include <nvbench/stopping_criterion.cuh>
#include <nvbench/types.cuh>

#include <cmath>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "test_asserts.cuh"

struct recording_printer : nvbench::printer_base
{
  explicit recording_printer(std::ostream &stream)
      : nvbench::printer_base{stream}
  {}

  std::vector<std::pair<nvbench::log_level, std::string>> logs;

protected:
  void do_log(nvbench::log_level level, const std::string &message) override
  {
    logs.emplace_back(level, message);
  }
};

void check_noise_warning(
  std::optional<nvbench::float64_t> stdev_noise,
  const std::string &expected_message,
  nvbench::int64_t total_samples = nvbench::detail::statistics::min_samples_for_noise_estimate)
{
  std::ostringstream stream;
  recording_printer printer{stream};
  nvbench::criterion_params params;
  params.set_float64("max-noise", 0.01);

  nvbench::detail::log_measurement_timeout_warnings(printer,
                                                    params,
                                                    1.0,
                                                    total_samples,
                                                    1,
                                                    1.0,
                                                    stdev_noise);

  ASSERT(printer.logs.size() == 1);
  ASSERT(printer.logs[0].first == nvbench::log_level::warn);
  ASSERT(printer.logs[0].second.find(expected_message) != std::string::npos);
}

void test_non_finite_or_invalid_stdev_noise_timeout_warning()
{
  check_noise_warning(std::nullopt,
                      "before accumulating enough samples to estimate noise",
                      nvbench::detail::statistics::min_samples_for_noise_estimate - 1);
  check_noise_warning(std::nullopt, "unable to estimate noise");
  check_noise_warning(std::numeric_limits<nvbench::float64_t>::quiet_NaN(),
                      "unable to estimate noise");
  check_noise_warning(-1.0, "unable to estimate noise");
  check_noise_warning(std::numeric_limits<nvbench::float64_t>::infinity(), "over noise threshold");
}

int main()
{
  test_non_finite_or_invalid_stdev_noise_timeout_warning();
  return 0;
}
