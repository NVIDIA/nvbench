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
#include <nvbench/types.cuh>

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <limits>
#include <random>
#include <sstream>
#include <vector>

#include "test_asserts.cuh"

namespace statistics = nvbench::detail::statistics;

inline constexpr nvbench::float64_t default_atol = 1.0e-14;
inline constexpr nvbench::float64_t default_rtol = 1.0e-14;

inline bool is_close(nvbench::float64_t actual,
                     nvbench::float64_t expected,
                     nvbench::float64_t atol,
                     nvbench::float64_t rtol)
{
  return std::abs(actual - expected) < std::max(atol, rtol * std::abs(expected));
}

inline bool is_close(nvbench::float64_t actual, nvbench::float64_t expected)
{
  return is_close(actual, expected, default_atol, default_rtol);
}

template <typename T>
void assert_quartiles_equal(statistics::quartiles_t<T> actual, statistics::quartiles_t<T> expected)
{
  ASSERT(actual.first_quartile == expected.first_quartile);
  ASSERT(actual.median == expected.median);
  ASSERT(actual.third_quartile == expected.third_quartile);
}

template <typename T>
void assert_quartiles_nan(statistics::quartiles_t<T> actual)
{
  ASSERT(std::isnan(actual.first_quartile));
  ASSERT(std::isnan(actual.median));
  ASSERT(std::isnan(actual.third_quartile));
}

void test_mean()
{
  {
    std::vector<nvbench::float64_t> data{1.0, 2.0, 3.0, 4.0, 5.0};
    const nvbench::float64_t actual   = statistics::compute_mean(std::begin(data), std::end(data));
    const nvbench::float64_t expected = 3.0;
    ASSERT(is_close(actual, expected));
  }

  {
    std::vector<nvbench::float64_t> data;
    const bool finite = std::isfinite(statistics::compute_mean(std::begin(data), std::end(data)));
    ASSERT(!finite);
  }
}

void test_online_mean_variance()
{
  {
    statistics::online_mean_variance stats;
    ASSERT(stats.get_size() == 0);
    ASSERT(stats.get_mean() == 0.0);
    ASSERT(stats.get_sample_variance() == 0.0);
    ASSERT(std::isnan(stats.get_unbiased_variance()));
  }

  {
    statistics::online_mean_variance stats;
    stats.update(42.0);

    ASSERT(stats.get_size() == 1);
    ASSERT(stats.get_mean() == 42.0);
    ASSERT(stats.get_sample_variance() == 0.0);
    ASSERT(std::isnan(stats.get_unbiased_variance()));
  }

  {
    statistics::online_mean_variance stats;
    for (const auto value : std::vector<nvbench::float64_t>{1.0, 2.0, 3.0, 4.0, 5.0})
    {
      stats.update(value);
    }

    ASSERT(stats.get_size() == 5);
    ASSERT(is_close(stats.get_mean(), 3.0));
    ASSERT(is_close(stats.get_sample_variance(), 2.0));
    ASSERT(is_close(stats.get_unbiased_variance(), 2.5));
  }

  {
    statistics::online_mean_variance left;
    left.update(1.0);

    statistics::online_mean_variance right;
    right.update(3.0);

    left.merge(right);

    ASSERT(left.get_size() == 2);
    ASSERT(left.get_mean() == 2.0);
    ASSERT(left.get_sample_variance() == 1.0);
    ASSERT(left.get_unbiased_variance() == 2.0);
  }

  {
    statistics::online_mean_variance left;
    left.update(1.0);
    left.update(2.0);

    statistics::online_mean_variance right;
    right.update(3.0);
    right.update(4.0);
    right.update(5.0);

    statistics::online_mean_variance merged = left;
    merged.merge(right);

    statistics::online_mean_variance expected;
    for (const auto value : std::vector<nvbench::float64_t>{1.0, 2.0, 3.0, 4.0, 5.0})
    {
      expected.update(value);
    }

    ASSERT(merged.get_size() == expected.get_size());
    ASSERT(is_close(merged.get_mean(), expected.get_mean()));
    ASSERT(is_close(merged.get_sample_variance(), expected.get_sample_variance()));
    ASSERT(is_close(merged.get_unbiased_variance(), expected.get_unbiased_variance()));
  }

  {
    statistics::online_mean_variance empty;
    statistics::online_mean_variance stats;
    stats.update(1.0);
    stats.update(3.0);

    const auto size              = stats.get_size();
    const auto mean              = stats.get_mean();
    const auto sample_variance   = stats.get_sample_variance();
    const auto unbiased_variance = stats.get_unbiased_variance();

    stats.merge(empty);

    ASSERT(stats.get_size() == size);
    ASSERT(stats.get_mean() == mean);
    ASSERT(stats.get_sample_variance() == sample_variance);
    ASSERT(stats.get_unbiased_variance() == unbiased_variance);
  }

  {
    statistics::online_mean_variance stats;
    stats.update(1.4e154);
    stats.update(1.4e154);

    statistics::online_mean_variance merged;
    merged.merge(stats);

    ASSERT(merged.get_size() == stats.get_size());
    ASSERT(merged.get_mean() == stats.get_mean());
    ASSERT(merged.get_sample_variance() == stats.get_sample_variance());
    ASSERT(merged.get_unbiased_variance() == stats.get_unbiased_variance());
  }
}

void test_std()
{
  {
    std::vector<nvbench::float64_t> data{1.0, 2.0, 3.0, 4.0, 5.0};
    const nvbench::float64_t mean = 3.0;
    const nvbench::float64_t actual =
      statistics::standard_deviation(std::begin(data), std::end(data), mean);
    const nvbench::float64_t expected = 1.581;
    ASSERT(is_close(actual, expected, 0.001, 0.0));
  }

  {
    std::vector<nvbench::float64_t> data;
    data.resize(static_cast<std::size_t>(statistics::min_samples_for_noise_estimate - 1), 1.0);
    const nvbench::float64_t actual =
      statistics::standard_deviation(std::begin(data), std::end(data), 1.0);
    ASSERT(!std::isfinite(actual));
  }
}

void test_percentiles()
{
  {
    const std::vector<nvbench::float64_t> data{40.0, 10.0, 30.0, 20.0};
    const auto actual = statistics::compute_percentiles(data.cbegin(),
                                                        data.cend(),
                                                        std::array<int, 5>{0, 25, 50, 75, 100});
    const std::array<nvbench::float64_t, 5> expected{10.0, 20.0, 30.0, 30.0, 40.0};
    ASSERT(actual == expected);
  }

  {
    const std::vector<nvbench::float64_t> data{42.0};
    const auto actual =
      statistics::compute_percentiles(data.cbegin(), data.cend(), std::array<int, 3>{25, 50, 75});
    const std::array<nvbench::float64_t, 3> expected{42.0, 42.0, 42.0};
    ASSERT(actual == expected);
  }

  {
    const std::vector<nvbench::float64_t> data{40.0, 10.0, 30.0, 20.0};
    const auto actual =
      statistics::compute_percentiles(data.cbegin(), data.cend(), std::array<int, 3>{25, 50, 75});
    const std::array<nvbench::float64_t, 3> expected{20.0, 30.0, 30.0};
    ASSERT(actual == expected);
  }

  {
    std::istringstream data{"40 10 30 20"};
    const auto actual =
      statistics::compute_percentiles(std::istream_iterator<nvbench::float64_t>{data},
                                      std::istream_iterator<nvbench::float64_t>{},
                                      std::array<int, 3>{25, 50, 75});
    const std::array<nvbench::float64_t, 3> expected{20.0, 30.0, 30.0};
    ASSERT(actual == expected);
  }

  {
    const std::vector<nvbench::float64_t> data{10.0, 20.0, 30.0, 40.0};
    const auto actual =
      statistics::compute_percentiles(data.cbegin(), data.cend(), std::array<int, 2>{-25, 125});
    const std::array<nvbench::float64_t, 2> expected{10.0, 40.0};
    ASSERT(actual == expected);
  }

  {
    const std::vector<nvbench::float64_t> data;
    const auto actual =
      statistics::compute_percentiles(data.cbegin(), data.cend(), std::array<int, 3>{25, 50, 75});
    ASSERT(std::isnan(actual[0]));
    ASSERT(std::isnan(actual[1]));
    ASSERT(std::isnan(actual[2]));
  }
}

void test_quartiles_methods_agree()
{
  {
    const std::vector<nvbench::float64_t> data{40.0, 10.0, 30.0, 20.0};
    const auto sorting =
      statistics::compute_quartiles_by_sorting(std::vector<nvbench::float64_t>(data));
    const auto selection =
      statistics::compute_quartiles_by_selection(std::vector<nvbench::float64_t>(data));
    assert_quartiles_equal(selection, sorting);
    assert_quartiles_equal(sorting, statistics::quartiles_t<nvbench::float64_t>{20.0, 30.0, 30.0});
  }

  {
    const std::vector<nvbench::float64_t> data{5.0, -1.0, 5.0, 2.0, 9.0, 2.0, 5.0};
    const auto sorting =
      statistics::compute_quartiles_by_sorting(std::vector<nvbench::float64_t>(data));
    const auto selection =
      statistics::compute_quartiles_by_selection(std::vector<nvbench::float64_t>(data));
    assert_quartiles_equal(selection, sorting);
  }

  // test around threshold when public API switches between implementations
  for (const auto n : std::array<std::size_t, 3>{4095, 4096, 4097})
  {
    std::vector<nvbench::float64_t> data(n);
    for (std::size_t i = 0; i < data.size(); ++i)
    {
      data[i] = static_cast<nvbench::float64_t>((i * 37) % data.size());
    }

    const auto public_api = statistics::compute_quartiles(data.cbegin(), data.cend());
    const auto sorting =
      statistics::compute_quartiles_by_sorting(std::vector<nvbench::float64_t>(data));
    const auto selection =
      statistics::compute_quartiles_by_selection(std::vector<nvbench::float64_t>(data));
    assert_quartiles_equal(selection, sorting);
    assert_quartiles_equal(public_api, sorting);
  }
}

void test_quartiles_methods_agree_with_duplicate_heavy_inputs()
{
  // Test around threshold when public API switches between implementations.
  for (const auto n : std::array<std::size_t, 3>{4095, 4096, 4097})
  {
    for (const auto seed : std::array<unsigned int, 3>{17u, 12345u, 987654321u})
    {
      std::vector<nvbench::float64_t> data(n);
      for (std::size_t i = 0; i < data.size(); ++i)
      {
        data[i] = static_cast<nvbench::float64_t>((4 * i) / data.size());
      }

      std::mt19937 rng{seed};
      std::shuffle(data.begin(), data.end(), rng);

      const auto public_api = statistics::compute_quartiles(data.cbegin(), data.cend());
      const auto sorting =
        statistics::compute_quartiles_by_sorting(std::vector<nvbench::float64_t>(data));
      const auto selection =
        statistics::compute_quartiles_by_selection(std::vector<nvbench::float64_t>(data));
      assert_quartiles_equal(selection, sorting);
      assert_quartiles_equal(public_api, sorting);
    }
  }
}

void test_quartiles()
{
  // special case inputs produce expected results
  {
    const std::vector<nvbench::float64_t> data{42.0};
    assert_quartiles_equal(statistics::compute_quartiles(data.cbegin(), data.cend()),
                           statistics::quartiles_t<nvbench::float64_t>{42.0, 42.0, 42.0});
  }

  {
    const std::vector<nvbench::float64_t> data;
    assert_quartiles_nan(statistics::compute_quartiles(data.cbegin(), data.cend()));
  }

  // works with input iterators
  {
    std::istringstream data{"40 10 30 20"};
    const auto actual =
      statistics::compute_quartiles(std::istream_iterator<nvbench::float64_t>{data},
                                    std::istream_iterator<nvbench::float64_t>{});
    assert_quartiles_equal(actual, statistics::quartiles_t<nvbench::float64_t>{20.0, 30.0, 30.0});
  }
}

void test_compute_relative_dispersion_nominal_input()
{
  {
    const auto actual = statistics::compute_relative_dispersion(6.0, 3.0);
    ASSERT(actual);
    ASSERT(is_close(*actual, 2.0));
  }

  // infinite dispersion is tolerated
  {
    const auto actual =
      statistics::compute_relative_dispersion(std::numeric_limits<nvbench::float64_t>::infinity(),
                                              1.0);
    ASSERT(actual);
    ASSERT(!std::isfinite(*actual));
  }
}

void test_compute_relative_dispersion_invalid_inputs()
{
  {
    const auto actual = statistics::compute_relative_dispersion(1.0, 0.0);
    ASSERT(!actual);
  }

  {
    const auto actual = statistics::compute_relative_dispersion(1.0, -1.0);
    ASSERT(!actual);
  }

  {
    const auto actual =
      statistics::compute_relative_dispersion(std::numeric_limits<nvbench::float64_t>::quiet_NaN(),
                                              1.0);
    ASSERT(!actual);
  }

  {
    const auto actual = statistics::compute_relative_dispersion(-1.0, 1.0);
    ASSERT(!actual);
  }
}

void test_compute_robust_noise()
{
  {
    const auto actual =
      statistics::compute_robust_noise(statistics::min_samples_for_noise_estimate - 1,
                                       2.0,
                                       4.0,
                                       6.0);
    ASSERT(!actual);
  }

  {
    const auto actual =
      statistics::compute_robust_noise(statistics::min_samples_for_noise_estimate, 2.0, 4.0, 6.0);
    ASSERT(actual);
    ASSERT(is_close(*actual, 1.0));
  }
}

void test_relative_interquartile_range()
{
  {
    const auto actual = statistics::compute_relative_interquartile_range(2.0, 4.0, 6.0);
    ASSERT(actual);
    ASSERT(is_close(*actual, 1.0));
  }

  {
    const auto actual = statistics::compute_relative_interquartile_range(0.0, 0.0, 1.0);
    ASSERT(!actual);
  }

  {
    const auto actual = statistics::compute_relative_interquartile_range(
      0.0,
      1.0,
      std::numeric_limits<nvbench::float64_t>::infinity());
    ASSERT(!actual);
  }

  {
    const auto actual = statistics::compute_relative_interquartile_range(
      0.0,
      std::numeric_limits<nvbench::float64_t>::min(),
      std::numeric_limits<nvbench::float64_t>::max());
    ASSERT(actual);
    ASSERT(!std::isfinite(*actual));
  }
}

void test_lin_regression()
{
  {
    std::vector<nvbench::float64_t> ys{1.0, 2.0, 3.0, 4.0, 5.0};
    auto [slope, intercept] = statistics::compute_linear_regression(std::begin(ys), std::end(ys));
    ASSERT(slope == 1.0);
    ASSERT(intercept == 1.0);
  }
  {
    std::vector<nvbench::float64_t> ys{42.0, 42.0, 42.0};
    auto [slope, intercept] = statistics::compute_linear_regression(std::begin(ys), std::end(ys));
    ASSERT(slope == 0.0);
    ASSERT(intercept == 42.0);
  }
  {
    std::vector<nvbench::float64_t> ys{8.0, 4.0, 0.0};
    auto [slope, intercept] = statistics::compute_linear_regression(std::begin(ys), std::end(ys));
    ASSERT(slope == -4.0);
    ASSERT(intercept == 8.0);
  }
}

void test_r2()
{
  {
    std::vector<nvbench::float64_t> ys{1.0, 2.0, 3.0, 4.0, 5.0};
    auto [slope, intercept] = statistics::compute_linear_regression(std::begin(ys), std::end(ys));
    const nvbench::float64_t actual =
      statistics::compute_r2(std::begin(ys), std::end(ys), slope, intercept);
    const nvbench::float64_t expected = 1.0;
    ASSERT(is_close(actual, expected, 0.001, 0.0));
  }
  {
    std::vector<nvbench::float64_t> signal{1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<nvbench::float64_t> noise{-1.0, 1.0, -1.0, 1.0, -1.0};
    std::vector<nvbench::float64_t> ys(signal.size());

    std::transform(std::begin(signal),
                   std::end(signal),
                   std::begin(noise),
                   std::begin(ys),
                   std::plus<nvbench::float64_t>());

    auto [slope, intercept] = statistics::compute_linear_regression(std::begin(ys), std::end(ys));
    const nvbench::float64_t expected = 0.675;
    const nvbench::float64_t actual =
      statistics::compute_r2(std::begin(ys), std::end(ys), slope, intercept);
    ASSERT(is_close(actual, expected, 0.001, 0.0));
  }
}

void test_slope_conversion()
{
  {
    const nvbench::float64_t actual   = statistics::slope2deg(0.0);
    const nvbench::float64_t expected = 0.0;
    ASSERT(is_close(actual, expected, 0.001, 0.0));
  }
  {
    const nvbench::float64_t actual   = statistics::slope2deg(1.0);
    const nvbench::float64_t expected = 45.0;
    ASSERT(is_close(actual, expected, 0.001, 0.0));
  }
  {
    const nvbench::float64_t actual   = statistics::slope2deg(5.0);
    const nvbench::float64_t expected = 78.69;
    ASSERT(is_close(actual, expected, 0.001, 0.0));
  }
}

int main()
{
  test_mean();
  test_online_mean_variance();
  test_std();
  test_percentiles();
  test_quartiles();
  test_quartiles_methods_agree();
  test_quartiles_methods_agree_with_duplicate_heavy_inputs();
  test_compute_relative_dispersion_nominal_input();
  test_compute_relative_dispersion_invalid_inputs();
  test_relative_interquartile_range();
  test_compute_robust_noise();
  test_lin_regression();
  test_r2();
  test_slope_conversion();
}
