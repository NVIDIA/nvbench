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

struct iteration_space_base
{
  using axes_type = std::vector<std::unique_ptr<nvbench::axis_base>>;
  using axes_info = std::vector<detail::axis_index>;

  using AdvanceSignature =
    nvbench::detail::axis_space_iterator::AdvanceSignature;
  using UpdateSignature = nvbench::detail::axis_space_iterator::UpdateSignature;

  iteration_space_base(std::vector<std::size_t> input_indices,
                  std::vector<std::size_t> output_indices);
  virtual ~iteration_space_base();

  [[nodiscard]] std::unique_ptr<iteration_space_base> clone() const;
  [[nodiscard]] std::vector<std::unique_ptr<iteration_space_base>>
  clone_as_linear() const;

  [[nodiscard]] detail::axis_space_iterator get_iterator(const axes_type &axes) const;
  [[nodiscard]] std::size_t get_size(const axes_type &axes) const;
  [[nodiscard]] std::size_t get_active_count(const axes_type &axes) const;

  [[nodiscard]] bool contains(std::size_t input_index) const;

protected:
  std::vector<std::size_t> m_input_indices;
  std::vector<std::size_t> m_output_indices;

  virtual std::unique_ptr<iteration_space_base> do_clone() const         = 0;
  virtual detail::axis_space_iterator do_get_iterator(axes_info info) const = 0;
  virtual std::size_t do_get_size(const axes_info &info) const          = 0;
  virtual std::size_t do_get_active_count(const axes_info &info) const   = 0;
};

struct linear_axis_space final : iteration_space_base
{
  linear_axis_space(std::size_t in, std::size_t out);
  ~linear_axis_space();

  std::unique_ptr<iteration_space_base> do_clone() const override;
  detail::axis_space_iterator do_get_iterator(axes_info info) const override;
  std::size_t do_get_size(const axes_info &info) const override;
  std::size_t do_get_active_count(const axes_info &info) const override;
};

struct zip_axis_space final : iteration_space_base
{
  zip_axis_space(std::vector<std::size_t> input_indices,
      std::vector<std::size_t> output_indices);
  ~zip_axis_space();

  std::unique_ptr<iteration_space_base> do_clone() const override;
  detail::axis_space_iterator do_get_iterator(axes_info info) const override;
  std::size_t do_get_size(const axes_info &info) const override;
  std::size_t do_get_active_count(const axes_info &info) const override;
};

struct user_axis_space : iteration_space_base
{
  user_axis_space(std::vector<std::size_t> input_indices,
                  std::vector<std::size_t> output_indices);
  ~user_axis_space();
};

using make_user_space_signature =
  std::unique_ptr<iteration_space_base>(std::vector<std::size_t> input_indices,
                                   std::vector<std::size_t> output_indices);

} // namespace nvbench
