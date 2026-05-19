/*
 *  Copyright 2021 NVIDIA Corporation
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

#include <nvbench/detail/transform_reduce.cuh>
#include <nvbench/types.cuh>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <optional>
#include <type_traits>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace nvbench::detail::statistics
{

inline constexpr nvbench::int64_t min_samples_for_noise_estimate = 5;

/**
 * Computes and returns the unbiased sample standard deviation.
 *
 * If the input has fewer than min_samples_for_noise_estimate samples, infinity is returned.
 */
template <typename Iter, typename ValueType = typename std::iterator_traits<Iter>::value_type>
ValueType standard_deviation(Iter first, Iter last, ValueType mean)
{
  static_assert(std::is_floating_point_v<ValueType>);

  const auto num = std::distance(first, last);

  if (num < min_samples_for_noise_estimate) // don't bother with low sample sizes.
  {
    return std::numeric_limits<ValueType>::infinity();
  }

  const auto variance = nvbench::detail::transform_reduce(first,
                                                          last,
                                                          ValueType{},
                                                          std::plus<>{},
                                                          [mean](auto val) {
                                                            val -= mean;
                                                            val *= val;
                                                            return val;
                                                          }) /
                        static_cast<ValueType>((num - 1)); // Bessel’s correction
  return std::sqrt(variance);
}

/**
 * Computes and returns the mean.
 *
 * If the input has fewer than 1 sample, infinity is returned.
 */
template <class It>
nvbench::float64_t compute_mean(It first, It last)
{
  const auto num = std::distance(first, last);

  if (num < 1)
  {
    return std::numeric_limits<nvbench::float64_t>::infinity();
  }

  return std::accumulate(first, last, 0.0) / static_cast<nvbench::float64_t>(num);
}

/**
 * Computes exact percentile values using rank round(p / 100 * (S - 1)).
 *
 * The input range is copied before sorting, so const iterators are supported.
 * If the input has fewer than 1 sample, all percentiles are returned as quiet NaNs.
 */
template <typename Iter,
          std::size_t N,
          typename ValueType = typename std::iterator_traits<Iter>::value_type>
std::array<ValueType, N> compute_percentiles(Iter first, Iter last, std::array<int, N> percentiles)
{
  static_assert(std::is_floating_point_v<ValueType>,
                "compute_percentiles requires a floating-point value type.");

  std::array<ValueType, N> result{};

  const auto num = std::distance(first, last);
  if (num < 1)
  {
    result.fill(std::numeric_limits<ValueType>::quiet_NaN());
    return result;
  }

  std::vector<ValueType> sorted(first, last);
  std::sort(sorted.begin(), sorted.end());

  const auto max_rank = static_cast<nvbench::float64_t>(sorted.size() - 1);
  for (std::size_t i = 0; i < N; ++i)
  {
    const auto clamped_percentile = std::clamp(percentiles[i], 0, 100);

    const auto quantile = static_cast<nvbench::float64_t>(clamped_percentile) / 100.0;
    const auto rank     = static_cast<std::size_t>(std::round(quantile * max_rank));

    result[i] = sorted[rank];
  }

  return result;
}

/**
 * Overload that supports calls like `compute_percentiles(first, last, {25, 50, 75})`.
 */
template <typename Iter,
          std::size_t N,
          typename ValueType = typename std::iterator_traits<Iter>::value_type>
std::array<ValueType, N> compute_percentiles(Iter first, Iter last, const int (&percentiles)[N])
{
  std::array<int, N> percentile_array{};
  std::copy(std::begin(percentiles), std::end(percentiles), percentile_array.begin());
  return compute_percentiles(first, last, percentile_array);
}

// Returns nullopt for invalid inputs. A +inf result is allowed: it represents
// unbounded relative dispersion rather than missing data.
inline std::optional<nvbench::float64_t> compute_relative_dispersion(nvbench::float64_t dispersion,
                                                                     nvbench::float64_t center)
{
  if (center <= nvbench::float64_t{} || !std::isfinite(center) ||
      dispersion < nvbench::float64_t{} || std::isnan(dispersion))
  {
    return std::nullopt;
  }

  return dispersion / center;
}

inline std::optional<nvbench::float64_t>
compute_relative_interquartile_range(nvbench::float64_t first_quartile,
                                     nvbench::float64_t median,
                                     nvbench::float64_t third_quartile)
{
  const auto interquartile_range = third_quartile - first_quartile;
  if (!std::isfinite(interquartile_range))
  {
    return std::nullopt;
  }

  return compute_relative_dispersion(interquartile_range, median);
}

/**
 * Computes linear regression and returns the slope and intercept
 *
 * This version takes precomputed mean of [first, last).
 * If the input has fewer than 2 samples, infinity is returned for both slope and intercept.
 */
template <class It>
std::pair<nvbench::float64_t, nvbench::float64_t>
compute_linear_regression(It first, It last, nvbench::float64_t mean_y)
{
  const std::size_t n = static_cast<std::size_t>(std::distance(first, last));

  if (n < 2)
  {
    return std::make_pair(std::numeric_limits<nvbench::float64_t>::infinity(),
                          std::numeric_limits<nvbench::float64_t>::infinity());
  }

  // Assuming x starts from 0
  const nvbench::float64_t mean_x = (static_cast<nvbench::float64_t>(n) - 1.0) / 2.0;

  // Calculate the numerator and denominator for the slope
  nvbench::float64_t numerator   = 0.0;
  nvbench::float64_t denominator = 0.0;

  for (std::size_t i = 0; i < n; ++i, ++first)
  {
    const nvbench::float64_t x_diff = static_cast<nvbench::float64_t>(i) - mean_x;
    numerator += x_diff * (*first - mean_y);
    denominator += x_diff * x_diff;
  }

  // Calculate the slope and intercept
  const nvbench::float64_t slope     = numerator / denominator;
  const nvbench::float64_t intercept = mean_y - slope * mean_x;

  return std::make_pair(slope, intercept);
}

/**
 * Computes linear regression and returns the slope and intercept
 *
 * If the input has fewer than 2 samples, infinity is returned for both slope and intercept.
 */
template <class It>
std::pair<nvbench::float64_t, nvbench::float64_t> compute_linear_regression(It first, It last)
{
  return compute_linear_regression(first, last, compute_mean(first, last));
}

/**
 * Computes and returns the R^2 (coefficient of determination)
 *
 * This version takes precomputed mean of [first, last).
 */
template <class It>
nvbench::float64_t compute_r2(It first,
                              It last,
                              nvbench::float64_t mean_y,
                              nvbench::float64_t slope,
                              nvbench::float64_t intercept)
{
  const std::size_t n = static_cast<std::size_t>(std::distance(first, last));

  nvbench::float64_t ss_tot = 0.0;
  nvbench::float64_t ss_res = 0.0;

  for (std::size_t i = 0; i < n; ++i, ++first)
  {
    const nvbench::float64_t y      = *first;
    const nvbench::float64_t y_pred = slope * static_cast<nvbench::float64_t>(i) + intercept;

    ss_tot += (y - mean_y) * (y - mean_y);
    ss_res += (y - y_pred) * (y - y_pred);
  }

  if (ss_tot == 0.0)
  {
    return 1.0;
  }

  return 1.0 - ss_res / ss_tot;
}

/**
 * Computes and returns the R^2 (coefficient of determination)
 */
template <class It>
nvbench::float64_t
compute_r2(It first, It last, nvbench::float64_t slope, nvbench::float64_t intercept)
{
  return compute_r2(first, last, compute_mean(first, last), slope, intercept);
}

inline nvbench::float64_t rad2deg(nvbench::float64_t rad) { return rad * 180.0 / M_PI; }

inline nvbench::float64_t slope2rad(nvbench::float64_t slope) { return std::atan2(slope, 1.0); }

inline nvbench::float64_t slope2deg(nvbench::float64_t slope) { return rad2deg(slope2rad(slope)); }

} // namespace nvbench::detail::statistics
