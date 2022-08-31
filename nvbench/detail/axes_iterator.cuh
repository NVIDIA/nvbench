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

struct axis_index
{
  axis_index() = default;

  explicit axis_index(const axis_base *axi)
      : index(0)
      , name(axi->get_name())
      , type(axi->get_type())
      , size(axi->get_size())
      , active_size(axi->get_size())
  {
    if (type == nvbench::axis_type::type)
    {
      active_size =
        static_cast<const nvbench::type_axis *>(axi)->get_active_count();
    }
  }
  std::size_t index;
  std::string name;
  nvbench::axis_type type;
  std::size_t size;
  std::size_t active_size;
};

struct axis_space_iterator
{
  using AdvanceSignature = bool(std::size_t &current_index, std::size_t length);
  using UpdateSignature  = void(std::size_t index,
                               std::vector<axis_index> &indices);

  axis_space_iterator(
    std::size_t axes_count,
    std::size_t iter_count,
    std::function<axis_space_iterator::AdvanceSignature> &&advance,
    std::function<axis_space_iterator::UpdateSignature> &&update)
      : m_number_of_axes(axes_count)
      , m_iteration_size(iter_count)
      , m_advance(std::move(advance))
      , m_update(std::move(update))
  {}

  axis_space_iterator(
    std::size_t axes_count,
    std::size_t iter_count,
    std::function<axis_space_iterator::UpdateSignature> &&update)
      : m_number_of_axes(axes_count)
      , m_iteration_size(iter_count)
      , m_update(std::move(update))
  {}

  [[nodiscard]] bool next()
  {
    return this->m_advance(m_current_index, m_iteration_size);
  }

  void update_indices(std::vector<axis_index> &indices) const
  {
    this->m_update(m_current_index, indices);
  }

  std::size_t m_number_of_axes              = 1;
  std::size_t m_iteration_size              = 1;
  std::function<AdvanceSignature> m_advance = [](std::size_t &current_index,
                                                 std::size_t length) {
    (current_index + 1 == length) ? current_index = 0 : current_index++;
    return (current_index == 0); // we rolled over
  };
  std::function<UpdateSignature> m_update = nullptr;

private:
  std::size_t m_current_index = 0;
};

} // namespace detail
} // namespace nvbench
