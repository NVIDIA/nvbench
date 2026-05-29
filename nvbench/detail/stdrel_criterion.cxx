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

#include <cmath> // std::sqrt

namespace nvbench::detail
{

// Allow transient invalid noise estimates while still terminating when
// stdev relative noise does not have finite value for this many
// consecutive iterations. The limit is high enough to tolerate short
// startup/transient phases, but bounded so a stream of invalid estimates
// cannot run until the wall-time timeout.
constexpr nvbench::int64_t invalid_noise_estimate_limit = 64;

stdrel_criterion::stdrel_criterion()
    : stopping_criterion_base{"stdrel",
                              {{"max-noise", 0.005}, // 0.5% stdrel
                               {"min-time", 0.5}}}   // 0.5 seconds
{}

void stdrel_criterion::do_initialize()
{
  m_measurements_summary                = {};
  m_consecutive_invalid_noise_estimates = 0;
  m_noise_tracker.clear();
}

void stdrel_criterion::do_add_measurement(nvbench::float64_t measurement)
{
  m_measurements_summary.update(measurement);

  if (m_measurements_summary.get_size() < statistics::min_samples_for_noise_estimate)
  {
    return;
  }

  // Compute convergence statistics using timing measurements
  // dispersion includes Bessel correction to preserve legacy behavior
  const auto unbiased_dispersion = std::sqrt(m_measurements_summary.get_unbiased_variance());
  const auto measurement_noise =
    statistics::compute_relative_dispersion(unbiased_dispersion, m_measurements_summary.get_mean());
  if (measurement_noise && std::isfinite(*measurement_noise))
  {
    m_consecutive_invalid_noise_estimates = 0;
    m_noise_tracker.push_back(*measurement_noise);
  }
  else
  {
    ++m_consecutive_invalid_noise_estimates;
  }
}

bool stdrel_criterion::do_is_finished()
{
  if (m_consecutive_invalid_noise_estimates >= invalid_noise_estimate_limit)
  {
    return true;
  }

  const auto total_measured_time =
    m_measurements_summary.get_mean() *
    static_cast<nvbench::float64_t>(m_measurements_summary.get_size());
  if (total_measured_time <= m_params.get_float64("min-time"))
  {
    return false;
  }

  if (m_noise_tracker.empty())
  {
    return false;
  }

  if (m_consecutive_invalid_noise_estimates != 0)
  {
    return false;
  }

  // Noise has dropped below threshold
  if (m_noise_tracker.back() < m_params.get_float64("max-noise"))
  {
    return true;
  }

  constexpr std::size_t min_noise_stability_window     = 64;
  constexpr std::size_t noise_stability_check_interval = 16;

  // Check if the noise has converged by inspecting a
  // trailing window of recorded noise measurements.
  // This helps identify benchmarks that are inherently noisy and would
  // never converge to the target noise threshold. This check ensures that the
  // benchmark will end if the noise stabilizes above the target threshold.
  // Gather some iterations before checking noise, and limit how often we
  // check this.
  const bool do_check_noise_stability =
    ((m_noise_tracker.size() > min_noise_stability_window) &&
     (m_measurements_summary.get_size() % noise_stability_check_interval == 0));
  if (do_check_noise_stability)
  {
    statistics::online_mean_variance noise_summary{};
    for (const auto noise_v : m_noise_tracker)
    {
      noise_summary.update(noise_v);
    }
    if (std::isfinite(noise_summary.get_mean()) &&
        std::isfinite(noise_summary.get_sample_variance()))
    {
      // If the rel stdev of the last N noise measurements is less than
      // 5%, consider the result stable.
      constexpr auto noise_threshold = 0.05;
      const auto mean_scaled         = noise_summary.get_mean() * noise_threshold;
      const auto variance_threshold  = mean_scaled * mean_scaled;
      if (noise_summary.get_sample_variance() < variance_threshold)
      {
        return true;
      }
    }
  }

  return false;
}

} // namespace nvbench::detail
