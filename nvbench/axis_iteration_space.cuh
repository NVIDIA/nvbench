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
 * Base class for all axi and axes iteration spaces.
 *
 * If we consider an axi to be a container of values, iteration_spaces
 * would be the different types of iterators supported by that container.
 *
 * With that in mind we get the following mapping:
 * * linear_axis_space is equivalant to a forward iterator.
 *
 * * zip_axis_space is equivalant to a zip iterator.
 *
 * * user_axis_space is equivalant to a transform iterator.
 *
 *
 */
struct iteration_space_base
{
  using axes_type = std::vector<std::unique_ptr<nvbench::axis_base>>;
  using axes_info = std::vector<detail::axis_index>;

  using AdvanceSignature =
    nvbench::detail::axis_space_iterator::AdvanceSignature;
  using UpdateSignature = nvbench::detail::axis_space_iterator::UpdateSignature;

  /*!
   * Construct a new iteration_space_base
   *
   * @param[input_indices]
   * @param[output_indices]
   */
  iteration_space_base(std::vector<std::size_t> input_indices,
                       std::vector<std::size_t> output_indices);
  virtual ~iteration_space_base();

  [[nodiscard]] std::unique_ptr<iteration_space_base> clone() const;
  [[nodiscard]] std::vector<std::unique_ptr<iteration_space_base>>
  clone_as_linear() const;

  /*!
   * Construct a new iteration_space_base
   *
   */
  [[nodiscard]] detail::axis_space_iterator
  get_iterator(const axes_type &axes) const;

  /*!
   * Construct a new iteration_space_base
   *
   */
  [[nodiscard]] std::size_t get_size(const axes_type &axes) const;

  /*!
   * Construct a new iteration_space_base
   *
   */
  [[nodiscard]] std::size_t get_active_count(const axes_type &axes) const;

  /*!
   * Construct a new iteration_space_base
   *
   */
  [[nodiscard]] bool contains(std::size_t input_index) const;

protected:
  std::vector<std::size_t> m_input_indices;
  std::vector<std::size_t> m_output_indices;

  virtual std::unique_ptr<iteration_space_base> do_clone() const            = 0;
  virtual detail::axis_space_iterator do_get_iterator(axes_info info) const = 0;
  virtual std::size_t do_get_size(const axes_info &info) const              = 0;
  virtual std::size_t do_get_active_count(const axes_info &info) const      = 0;
};

} // namespace nvbench
