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

#include <nvbench/axis_base.cuh>
#include <nvbench/type_axis.cuh>

#include <functional>
#include <utility>
#include <vector>

namespace nvbench
{
namespace detail
{

// Tracks current value and axis information used while iterating through axes.
struct axis_value_index
{
  axis_value_index() = default;

  explicit axis_value_index(const axis_base *axis)
      : value_index(0)
      , axis_name(axis->get_name())
      , axis_type(axis->get_type())
      , axis_size(axis->get_size())
      , axis_active_size(axis_type == nvbench::axis_type::type
                           ? static_cast<const nvbench::type_axis *>(axis)->get_active_count()
                           : axis->get_size())
  {}

  std::size_t value_index;
  std::string axis_name;
  nvbench::axis_type axis_type;
  std::size_t axis_size;
  std::size_t axis_active_size;
};

struct axis_space_iterator
{
  using axis_value_indices = std::vector<detail::axis_value_index>;
  using advance_signature  = bool(std::size_t &current_iteration, std::size_t iteration_size);
  using update_signature   = void(std::size_t current_iteration,
                                axis_value_indices::iterator start_axis_value_info,
                                axis_value_indices::iterator end_axis_value_info);

  axis_space_iterator(axis_value_indices info,
                      std::size_t iteration_size,
                      std::function<axis_space_iterator::advance_signature> &&advance,
                      std::function<axis_space_iterator::update_signature> &&update)
      : m_iteration_size(iteration_size)
      , m_axis_value_indices(std::move(info))
      , m_advance(std::move(advance))
      , m_update(std::move(update))
  {}

  axis_space_iterator(axis_value_indices info,
                      std::size_t iter_count,
                      std::function<axis_space_iterator::update_signature> &&update)
      : m_iteration_size(iter_count)
      , m_axis_value_indices(std::move(info))
      , m_update(std::move(update))
  {}

  [[nodiscard]] bool next() { return m_advance(m_current_iteration, m_iteration_size); }

  void update_axis_value_indices(axis_value_indices &info) const
  {
    using diff_t = typename axis_value_indices::difference_type;
    info.insert(info.end(), m_axis_value_indices.begin(), m_axis_value_indices.end());
    axis_value_indices::iterator end   = info.end();
    axis_value_indices::iterator start = end - static_cast<diff_t>(m_axis_value_indices.size());
    m_update(m_current_iteration, start, end);
  }

  [[nodiscard]] const axis_value_indices &get_axis_value_indices() const
  {
    return m_axis_value_indices;
  }
  [[nodiscard]] axis_value_indices &get_axis_value_indices() { return m_axis_value_indices; }

  [[nodiscard]] std::size_t get_iteration_size() const { return m_iteration_size; }

private:
  std::size_t m_current_iteration = 0;
  std::size_t m_iteration_size    = 1;

  axis_value_indices m_axis_value_indices;

  std::function<advance_signature> m_advance = [](std::size_t &current_iteration,
                                                  std::size_t iteration_size) {
    (current_iteration + 1 == iteration_size) ? current_iteration = 0 : current_iteration++;
    return (current_iteration == 0); // we rolled over
  };

  std::function<update_signature> m_update = nullptr;
};

} // namespace detail
} // namespace nvbench
