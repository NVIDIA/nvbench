/*
 *  Copyright 2025 NVIDIA Corporation
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

#include <nvbench/types.cuh>

#include <cmath>
#include <limits>
#include <utility>

namespace nvbench::detail
{

class online_linear_regression
{
  nvbench::float64_t m_sum_x{};
  nvbench::float64_t m_sum_y{};
  nvbench::float64_t m_sum_xy{};
  nvbench::float64_t m_sum_x2{};
  nvbench::float64_t m_sum_y2{};
  nvbench::int64_t m_count{};

public:
  online_linear_regression() = default;

  void update(std::pair<nvbench::float64_t, nvbench::float64_t> incoming)
  {
    const auto [x, y] = incoming;
    m_sum_x += x;
    m_sum_y += y;
    m_sum_xy += x * y;
    m_sum_x2 += x * x;
    m_sum_y2 += y * y;
    m_count++;
  }

  void update(std::pair<nvbench::float64_t, nvbench::float64_t> outgoing,
              std::pair<nvbench::float64_t, nvbench::float64_t> incoming)
  {
    const auto [x_out, y_out] = outgoing;
    m_sum_x -= x_out;
    m_sum_y -= y_out;
    m_sum_xy -= x_out * y_out;
    m_sum_x2 -= x_out * x_out;
    m_sum_y2 -= y_out * y_out;

    const auto [x_in, y_in] = incoming;
    m_sum_x += x_in;
    m_sum_y += y_in;
    m_sum_xy += x_in * y_in;
    m_sum_x2 += x_in * x_in;
    m_sum_y2 += y_in * y_in;
  }

  void slide_window(nvbench::float64_t y_out, nvbench::float64_t y_in)
  {
    m_sum_y -= y_out;
    m_sum_y += y_in;

    m_sum_y2 -= y_out * y_out;
    m_sum_y2 += y_in * y_in;

    m_sum_xy -= m_sum_y - y_in;
    m_sum_xy += (static_cast<nvbench::float64_t>(m_count) - 1.0) * y_in;
  }

  void clear()
  {
    m_sum_x  = 0.0;
    m_sum_y  = 0.0;
    m_sum_xy = 0.0;
    m_sum_x2 = 0.0;
    m_sum_y2 = 0.0;
    m_count  = 0;
  }

  [[nodiscard]] nvbench::int64_t count() const { return m_count; }

  [[nodiscard]] nvbench::float64_t mean_x() const
  {
    return m_count > 0 ? m_sum_x / static_cast<nvbench::float64_t>(m_count) : 0.0;
  }

  [[nodiscard]] nvbench::float64_t mean_y() const
  {
    return m_count > 0 ? m_sum_y / static_cast<nvbench::float64_t>(m_count) : 0.0;
  }

  [[nodiscard]] nvbench::float64_t slope() const
  {
    static constexpr nvbench::float64_t q_nan =
      std::numeric_limits<nvbench::float64_t>::quiet_NaN();

    if (m_count < 2)
      return q_nan;

    const nvbench::float64_t n      = static_cast<nvbench::float64_t>(m_count);
    const nvbench::float64_t mean_x = (m_sum_x / n);
    const nvbench::float64_t mean_y = (m_sum_y / n);

    const nvbench::float64_t numerator   = (m_sum_xy / n) - mean_x * mean_y;
    const nvbench::float64_t denominator = (m_sum_x2 / n) - mean_x * mean_x;

    if (std::abs(denominator) < 1e-12)
      return q_nan;

    return numerator / denominator;
  }

  [[nodiscard]] nvbench::float64_t intercept() const
  {
    if (m_count < 2)
    {
      return std::numeric_limits<nvbench::float64_t>::quiet_NaN();
    }

    const nvbench::float64_t current_slope = slope();

    if (!std::isfinite(current_slope))
    {
      return std::numeric_limits<nvbench::float64_t>::quiet_NaN();
    }

    return mean_y() - current_slope * mean_x();
  }

  [[nodiscard]] nvbench::float64_t r_squared() const
  {
    if (m_count < 2)
    {
      return std::numeric_limits<nvbench::float64_t>::quiet_NaN();
    }

    // ss_tot and ss_res scaled by 1/n to avoid overflow
    const nvbench::float64_t n        = static_cast<nvbench::float64_t>(m_count);
    const nvbench::float64_t mean_y_v = mean_y();
    const nvbench::float64_t ss_tot   = (m_sum_y2 / n) - mean_y_v * mean_y_v;

    if (ss_tot < std::numeric_limits<nvbench::float64_t>::epsilon())
    {
      return 1.0;
    }

    const nvbench::float64_t slope_v     = slope();
    const nvbench::float64_t intercept_v = intercept();

    if (!std::isfinite(slope_v) || !std::isfinite(intercept_v))
    {
      return std::numeric_limits<nvbench::float64_t>::quiet_NaN();
    }
    else
    {
      const nvbench::float64_t mean_xy_v = m_sum_xy / n;
      const nvbench::float64_t mean_xx_v = m_sum_x2 / n;
      const nvbench::float64_t mean_x_v  = m_sum_x / n;
      const nvbench::float64_t ss_tot_m_res =
        slope_v * ((mean_xy_v - slope_v * mean_xx_v) + (mean_xy_v - intercept_v * mean_x_v)) +
        intercept_v * (mean_y_v - slope_v * mean_x_v - intercept_v) +
        mean_y_v * (intercept_v - mean_y_v);

      return std::min(std::max(ss_tot_m_res / ss_tot, 0.0), 1.0);
    }
  }

  [[nodiscard]] nvbench::float64_t sum_x() const { return m_sum_x; }
  [[nodiscard]] nvbench::float64_t sum_y() const { return m_sum_y; }
  [[nodiscard]] nvbench::float64_t sum_xy() const { return m_sum_xy; }
  [[nodiscard]] nvbench::float64_t sum_x2() const { return m_sum_x2; }
  [[nodiscard]] nvbench::float64_t sum_y2() const { return m_sum_y2; }
};

} // namespace nvbench::detail
