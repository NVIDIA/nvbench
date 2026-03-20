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
  m_regression.clear();
}

void entropy_criterion::update_entropy_sum(nvbench::float64_t old_count,
                                           nvbench::float64_t new_count)
{
  if (old_count > 0)
  {
    auto diff = new_count - old_count;
    m_sum_count_log_counter += new_count * std::log2(1 + diff / old_count) +
                               diff * std::log2(old_count);
  }
  else
  {
    m_sum_count_log_counter += new_count * std::log2(new_count);
  }
}

nvbench::float64_t entropy_criterion::compute_entropy()
{
  if (m_total_samples == 0)
  {
    return 0.0;
  }

  const auto n                     = static_cast<nvbench::float64_t>(m_total_samples);
  const nvbench::float64_t entropy = std::log2(n) - m_sum_count_log_counter / n;

  return std::copysign(std::max(0.0, entropy), 1.0);
}

void entropy_criterion::do_add_measurement(nvbench::float64_t measurement)
{
  m_total_samples++;
  m_total_cuda_time += measurement;
  nvbench::int64_t old_count = 0;
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
    }
    else
    {
      old_count = 0;
      m_freq_tracker.insert(it, std::make_pair(key, nvbench::int64_t{1}));
    }
  }

  update_entropy_sum(static_cast<nvbench::float64_t>(old_count),
                     static_cast<nvbench::float64_t>(old_count + 1));
  const nvbench::float64_t entropy = compute_entropy();
  const nvbench::float64_t n       = static_cast<nvbench::float64_t>(m_entropy_tracker.size());

  if (m_entropy_tracker.size() == m_entropy_tracker.capacity())
  {
    const nvbench::float64_t old_entropy = *m_entropy_tracker.cbegin();

    m_regression.slide_window(old_entropy, entropy);
  }
  else
  {
    const nvbench::float64_t new_x = n;
    m_regression.update({new_x, entropy});
  }

  m_entropy_tracker.push_back(entropy);
}

bool entropy_criterion::do_is_finished()
{
  if (m_entropy_tracker.size() < 2)
  {
    return false;
  }

  if (m_total_samples % 2 != 0)
  {
    return false;
  }

  const nvbench::float64_t slope = m_regression.slope();

  if (!std::isfinite(slope))
  {
    return false;
  }

  if (statistics::slope2deg(slope) > m_params.get_float64("max-angle"))
  {
    return false;
  }

  const nvbench::float64_t r2 = m_regression.r_squared();

  if (!std::isfinite(r2))
  {
    return false;
  }

  if (r2 < m_params.get_float64("min-r2"))
  {
    return false;
  }

  return true;
}

} // namespace nvbench::detail
