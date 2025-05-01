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

#include <nvbench/float64_axis.cuh>
#include <nvbench/int64_axis.cuh>
#include <nvbench/iteration_space_base.cuh>
#include <nvbench/linear_axis_space.cuh>
#include <nvbench/string_axis.cuh>
#include <nvbench/type_axis.cuh>
#include <nvbench/types.cuh>
#include <nvbench/user_axis_space.cuh>
#include <nvbench/zip_axis_space.cuh>

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace nvbench
{

// Holds dynamic axes information.
struct axes_metadata
{
  using axes_type            = std::vector<std::unique_ptr<nvbench::axis_base>>;
  using iteration_space_type = std::vector<std::unique_ptr<nvbench::iteration_space_base>>;

  template <typename... TypeAxes>
  explicit axes_metadata(nvbench::type_list<TypeAxes...>);

  axes_metadata()                            = default;
  axes_metadata(axes_metadata &&)            = default;
  axes_metadata &operator=(axes_metadata &&) = default;

  axes_metadata(const axes_metadata &);
  axes_metadata &operator=(const axes_metadata &);

  void set_type_axes_names(std::vector<std::string> names);

  void add_int64_axis(std::string name,
                      std::vector<nvbench::int64_t> data,
                      nvbench::int64_axis_flags flags);

  void add_float64_axis(std::string name, std::vector<nvbench::float64_t> data);

  void add_string_axis(std::string name, std::vector<std::string> data);

  void add_axis(const axis_base &axis);

  template <typename... Args>
  void add_zip_axes(Args &&...args)
  {
    const std::size_t start = this->m_axes.size();
    const std::size_t count = sizeof...(Args);
    // (this->add_axis(std::forward<Args>(args)), ...);
    (m_axes.push_back(args.clone()), ...);
    this->add_zip_space(start, count);
  }

  template <typename... Args>
  void add_user_iteration_axes(std::function<nvbench::make_user_space_signature> make,
                               Args &&...args)
  {
    const std::size_t start = this->m_axes.size();
    const std::size_t count = sizeof...(Args);
    (m_axes.push_back(args.clone()), ...);
    this->add_user_iteration_space(std::move(make), start, count);
  }

  [[nodiscard]] const iteration_space_type &get_type_iteration_space() const
  {
    return m_type_space;
  }
  [[nodiscard]] const iteration_space_type &get_value_iteration_space() const
  {
    return m_value_space;
  }

  [[nodiscard]] const nvbench::int64_axis &get_int64_axis(std::string_view name) const;
  [[nodiscard]] nvbench::int64_axis &get_int64_axis(std::string_view name);

  [[nodiscard]] const nvbench::float64_axis &get_float64_axis(std::string_view name) const;
  [[nodiscard]] nvbench::float64_axis &get_float64_axis(std::string_view name);

  [[nodiscard]] const nvbench::string_axis &get_string_axis(std::string_view name) const;
  [[nodiscard]] nvbench::string_axis &get_string_axis(std::string_view name);

  [[nodiscard]] const nvbench::type_axis &get_type_axis(std::string_view name) const;
  [[nodiscard]] nvbench::type_axis &get_type_axis(std::string_view name);

  [[nodiscard]] const nvbench::type_axis &get_type_axis(std::size_t index) const;
  [[nodiscard]] nvbench::type_axis &get_type_axis(std::size_t index);

  [[nodiscard]] const axes_type &get_axes() const { return m_axes; }
  [[nodiscard]] axes_type &get_axes() { return m_axes; }

  [[nodiscard]] const nvbench::axis_base &get_axis(std::string_view name) const;
  [[nodiscard]] nvbench::axis_base &get_axis(std::string_view name);

  [[nodiscard]] const nvbench::axis_base &get_axis(std::string_view name,
                                                   nvbench::axis_type type) const;
  [[nodiscard]] nvbench::axis_base &get_axis(std::string_view name, nvbench::axis_type type);

  [[nodiscard]] static std::vector<std::string>
  generate_default_type_axis_names(std::size_t num_type_axes);

private:
  axes_type m_axes;
  std::size_t m_type_axe_count = 0;
  iteration_space_type m_type_space;
  iteration_space_type m_value_space;

  void add_zip_space(std::size_t first_index, std::size_t count);
  void add_user_iteration_space(std::function<nvbench::make_user_space_signature> make,
                                std::size_t first_index,
                                std::size_t count);
};

template <typename... TypeAxes>
axes_metadata::axes_metadata(nvbench::type_list<TypeAxes...>)
    : axes_metadata{}
{
  using type_axes_list         = nvbench::type_list<TypeAxes...>;
  constexpr auto num_type_axes = nvbench::tl::size<type_axes_list>::value;
  auto names                   = axes_metadata::generate_default_type_axis_names(num_type_axes);

  auto names_iter = names.begin(); // contents will be moved from
  nvbench::tl::foreach<type_axes_list>(
    [&axes = m_axes, &spaces = m_type_space, &names_iter]([[maybe_unused]] auto wrapped_type) {
      // This is always called before other axes are added, so the length of the
      // axes vector will be the type axis index:
      const std::size_t type_axis_index = axes.size();

      spaces.push_back(std::make_unique<linear_axis_space>(type_axis_index));

      // Note:
      // The word "type" appears 6 times in the next line.
      // Every. Single. Token.
      typedef typename decltype(wrapped_type)::type type_list;
      auto axis = std::make_unique<nvbench::type_axis>(std::move(*names_iter++), type_axis_index);
      axis->template set_inputs<type_list>();
      axes.push_back(std::move(axis));
    });
  m_type_axe_count = m_axes.size();
}

} // namespace nvbench
