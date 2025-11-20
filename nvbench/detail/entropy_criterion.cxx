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

#include <nvbench/detail/entropy_criterion.cuh>
#include <nvbench/types.cuh>

#include <cmath>

namespace nvbench::detail
{

entropy_criterion::entropy_criterion()
    : stopping_criterion_base{"entropy", {{"max-angle", 0.048}, {"min-r2", 0.36}}}
{
  m_freq_tracker.reserve(m_entropy_tracker.capacity() * 2);
}

void entropy_criterion::do_initialize()
{
  m_total_samples   = 0;
  m_total_cuda_time = 0.0;
  m_entropy_tracker.clear();
  m_freq_tracker.clear();

  m_sum_count_log_counter = 0.0;
  m_sum_x                 = 0.0;
  m_sum_y                 = 0.0;
  m_sum_x2                = 0.0;
  m_sum_y2                = 0.0;
  m_sum_xy                = 0.0;
}

void entropy_criterion::update_entropy_sum(nvbench::float64_t old_count,
                                           nvbench::float64_t new_count)
{
  if (old_count > 0)
  {
    m_sum_count_log_counter -= old_count * std::log2(old_count);
  }

  if (new_count > 0)
  {
    m_sum_count_log_counter += new_count * std::log2(new_count);
  }
}

nvbench::float64_t entropy_criterion::compute_entropy()
{
  const nvbench::size_t n = m_total_samples;

  if (n == 0)
  {
    return 0.0;
  }

  const nvbench::float64_t n_float = static_cast<nvbench::float64_t>(n);
  const nvbench::float64_t entropy = std::log2(n_float) - m_sum_count_log_counter / n_float;

  return (std::max(0.0, entropy) == 0.0) ? 0.0 : entropy;
}

void entropy_criterion::do_add_measurement(nvbench::float64_t measurement)
{
  m_total_samples++;
  m_total_cuda_time += measurement;
  nvbench::int64_t old_count = 0;
  nvbench::int64_t new_count = 0;
  {
    auto key                = measurement;
    constexpr bool bin_keys = false;

    if (bin_keys)
    {
      const auto resolution_us = 0.5;
      const auto resulution_s  = resolution_us / 1000000;
      const auto epsilon       = resulution_s * 2;
      key                      = std::round(key / epsilon) * epsilon;
    }

    // This approach is about 3x faster than `std::{unordered_,}map`
    // Up to 100k samples, only about 20% slower than corresponding stdrel method
    auto it = std::lower_bound(m_freq_tracker.begin(),
                               m_freq_tracker.end(),
                               std::make_pair(key, nvbench::int64_t{}));

    if (it != m_freq_tracker.end() && it->first == key)
    {
      old_count = it->second;
      it->second += 1;
      new_count = it->second;
    }
    else
    {
      old_count = 0;
      m_freq_tracker.insert(it, std::make_pair(key, nvbench::int64_t{1}));
      new_count = 1;
    }
  }

  update_entropy_sum(old_count, new_count);
  const nvbench::float64_t entropy = compute_entropy();
  const nvbench::float64_t n       = static_cast<nvbench::float64_t>(m_entropy_tracker.size() - 1);

  // remove oldest value from stats if buffer is full
  if (m_entropy_tracker.size() == m_entropy_tracker.capacity())
  {
    // Buffer is full, need to remove oldest entropy value
    const nvbench::float64_t old_entropy = *m_entropy_tracker.cbegin();
    const nvbench::float64_t old_x       = 0.0; // Oldest position in sliding window
    const nvbench::float64_t old_sum_x   = m_sum_x;

    // Remove old value contributions
    m_sum_x -= old_x + n;
    m_sum_y -= old_entropy;
    m_sum_xy -= old_x * old_entropy;
    m_sum_x2 -= old_x * old_x;
    m_sum_y2 -= old_entropy * old_entropy;

    // Adjust for X-index shift (all remaining indices shift down by 1)
    m_sum_x2 -= 2.0 * m_sum_x + n;
    m_sum_xy -= m_sum_y;
  }

  m_entropy_tracker.push_back(entropy);

  // Stat update
  const nvbench::float64_t x = n;
  const nvbench::float64_t y = entropy;

  m_sum_x += x;
  m_sum_y += y;
  m_sum_xy += x * y;
  m_sum_x2 += x * x;
  m_sum_y2 += y * y;
}

bool entropy_criterion::do_is_finished()
{
  if (m_entropy_tracker.size() < 2)
  {
    return false;
  }

  // Even number of samples is used to reduce the overhead and not required to compute entropy.
  // This makes `is_finished()` about 20% faster than corresponding stdrel method.
  if (m_total_samples % 2 != 0)
  {
    return false;
  }

  const nvbench::float64_t n      = static_cast<nvbench::float64_t>(m_entropy_tracker.size());
  const nvbench::float64_t mean_x = m_sum_x / n;
  const nvbench::float64_t mean_y = m_sum_y / n;

  const nvbench::float64_t numerator   = m_sum_xy - n * mean_x * mean_y;
  const nvbench::float64_t denominator = m_sum_x2 - n * mean_x * mean_x;
  if (denominator < 1e-9)
  {
    return false;
  }

  const nvbench::float64_t slope     = numerator / denominator;
  const nvbench::float64_t intercept = mean_y - slope * mean_x;

  if (statistics::slope2deg(slope) > m_params.get_float64("max-angle"))
  {
    return false;
  }

  nvbench::float64_t r2           = 0.0;
  const nvbench::float64_t ss_tot = m_sum_y2 - n * mean_y * mean_y;

  // When there's no variance in the data (perfect horizontal line), R² = 1
  if (ss_tot < 1e-9)
  {
    r2 = 1.0;
  }
  else
  {
    const nvbench::float64_t ss_res = m_sum_y2 - 2.0 * slope * m_sum_xy -
                                      2.0 * intercept * m_sum_y + slope * slope * m_sum_x2 +
                                      2.0 * slope * intercept * m_sum_x + n * intercept * intercept;
    r2 = std::max(0.0, std::min(1.0, 1.0 - (ss_res / ss_tot)));
  }

  if (r2 < m_params.get_float64("min-r2"))
  {
    return false;
  }

  return true;
}

} // namespace nvbench::detail
