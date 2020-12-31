#pragma once

#include <nvbench/float64_axis.cuh>
#include <nvbench/int64_axis.cuh>
#include <nvbench/string_axis.cuh>
#include <nvbench/type_axis.cuh>
#include <nvbench/types.cuh>

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
  using axes_type = std::vector<std::unique_ptr<nvbench::axis_base>>;

  template <typename type_axes>
  void set_type_axes_names(std::vector<std::string> names);

  void add_int64_axis(std::string name,
                      std::vector<nvbench::int64_t> data,
                      nvbench::int64_axis_flags flags);

  void add_float64_axis(std::string name, std::vector<nvbench::float64_t> data);

  void add_string_axis(std::string name, std::vector<std::string> data);

  [[nodiscard]] const nvbench::int64_axis &
  get_int64_axis(std::string_view name) const;

  [[nodiscard]] const nvbench::float64_axis &
  get_float64_axis(std::string_view name) const;

  [[nodiscard]] const nvbench::string_axis &
  get_string_axis(std::string_view name) const;

  [[nodiscard]] const nvbench::type_axis &
  get_type_axis(std::string_view name) const;

  [[nodiscard]] const axes_type &get_axes() const { return m_axes; }

  [[nodiscard]] const nvbench::axis_base &get_axis(std::string_view name) const;

  [[nodiscard]] const nvbench::axis_base &
  get_axis(std::string_view name, nvbench::axis_type type) const;

  [[nodiscard]] const nvbench::type_axis &get_type_axis(std::size_t index) const;

private:
  axes_type m_axes;
};

template <typename type_axes>
void axes_metadata::set_type_axes_names(std::vector<std::string> names)
{
  if (names.size() != nvbench::tl::size<type_axes>::value)
  { // TODO Find a way to get a better error message w/o bringing fmt
    // into this header.
    throw std::runtime_error("set_type_axes_names(): len(names) != "
                             "len(type_axes)");
  }
  std::size_t axis_index = 0;
  auto names_iter        = names.begin(); // contents will be moved from
  nvbench::tl::foreach<type_axes>([&axes = m_axes, &names_iter, &axis_index](
                                    [[maybe_unused]] auto wrapped_type) {
    // Note:
    // The word "type" appears 6 times in the next line.
    // Every. Single. Token.
    typedef typename decltype(wrapped_type)::type type_list;
    auto axis = std::make_unique<nvbench::type_axis>(std::move(*names_iter++),
                                                     axis_index++);
    axis->set_inputs<type_list>();
    axes.push_back(std::move(axis));
  });
}

} // namespace nvbench
