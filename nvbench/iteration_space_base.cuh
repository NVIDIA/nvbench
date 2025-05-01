/*
 *  Copyright 2022 NVIDIA Corporation
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

#include <nvbench/detail/axes_iterator.cuh>

namespace nvbench
{

/*!
 * Base class for all axis iteration spaces.
 *
 * If we consider an axis to be a container of values, iteration_spaces
 * would be how we can create iterators over that container.
 *
 * With that in mind we get the following mapping:
 * * linear_axis_space is equivalent to a forward iterator.
 *
 * * zip_axis_space is equivalent to a zip iterator.
 *
 * * user_axis_space is equivalent to a transform iterator.
 *
 * The `nvbench::axes_metadata` stores all axes in a std::vector. To represent
 * which axes each space is 'over' we store those indices. We don't store
 * the pointers or names for the following reasons:
 *
 * * The names of an axis can change after being added. The `nvbench::axes_metadata`
 * is not aware of the name change, and can't inform this class of it.
 *
 * * The `nvbench::axes_metadata` can be deep copied, which would invalidate
 * any pointers held by this class. By holding onto the index we remove the need
 * to do any form of fixup on deep copies of `nvbench::axes_metadata`.
 *
 *
 */
struct iteration_space_base
{
  using axes_type = std::vector<std::unique_ptr<nvbench::axis_base>>;
  using axes_info = std::vector<detail::axis_index>;

  using AdvanceSignature = nvbench::detail::axis_space_iterator::AdvanceSignature;
  using UpdateSignature  = nvbench::detail::axis_space_iterator::UpdateSignature;

  /*!
   * Construct a new derived iteration_space
   *
   * The input_indices and output_indices combine together to allow the iteration space to know
   * what axes they should query from axes_metadata and where each of those map to in the output
   * iteration space.
   * @param[input_indices] recorded indices of each axis from the axes metadata value space
   */
  iteration_space_base(std::vector<std::size_t> input_indices);
  virtual ~iteration_space_base();

  [[nodiscard]] std::unique_ptr<iteration_space_base> clone() const;

  /*!
   * Returns the iterator over the @a axis provided
   *
   * @param[axes]
   *
   */
  [[nodiscard]] detail::axis_space_iterator get_iterator(const axes_type &axes) const;

  /*!
   * Returns the number of active and inactive elements the iterator will have
   * when executed over @a axes
   *
   * Note:
   *  Type Axis support inactive elements
   */
  [[nodiscard]] std::size_t get_size(const axes_type &axes) const;

  /*!
   * Returns the number of active elements the iterator will have when
   * executed over @a axes
   *
   * Note:
   *  Type Axis support inactive elements
   */
  [[nodiscard]] std::size_t get_active_count(const axes_type &axes) const;

protected:
  std::vector<std::size_t> m_input_indices;

  virtual std::unique_ptr<iteration_space_base> do_clone() const            = 0;
  virtual detail::axis_space_iterator do_get_iterator(axes_info info) const = 0;
  virtual std::size_t do_get_size(const axes_info &info) const              = 0;
  virtual std::size_t do_get_active_count(const axes_info &info) const      = 0;
};

} // namespace nvbench
