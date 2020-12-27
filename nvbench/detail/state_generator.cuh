#pragma once

#include <nvbench/axis_base.cuh> // for axis_type

#include <string_view>
#include <vector>

namespace nvbench
{

namespace detail
{

struct state_generator
{
  struct axis_index
  {
    std::string_view axis;
    nvbench::axis_type type;
    std::size_t index;
    std::size_t size;
  };

  void add_axis(const nvbench::axis_base &axis)
  {
    this->add_axis(axis.get_name(), axis.get_type(), axis.get_size());
  }

  void add_axis(std::string_view axis,
                nvbench::axis_type type,
                std::size_t size)
  {
    m_indices.push_back({std::move(axis), type, std::size_t{0}, size});
  }

  [[nodiscard]] std::size_t get_number_of_states() const;

  // Yep, this class is its own non-STL-style iterator.
  // It's fiiiiine, we're in detail::. PRs welcome.
  //
  // Usage:
  // ```
  // state_generator sg;
  // sg.add_axis(...);
  // for (sg.init(); sg.iter_valid(); sg.next())
  // {
  //   for (const auto& axis_index : sg.get_current_indices())
  //   {
  //     std::string axis_name = axis_index.axis;
  //     nvbench::axis_type type = axis_index.type;
  //     std::size_t value_index = axis_index.index;
  //   }
  // }
  // ```
  void init();
  [[nodiscard]] const std::vector<axis_index> &get_current_indices()
  {
    return m_indices;
  }
  [[nodiscard]] bool iter_valid() const;
  void next();

private:
  std::vector<axis_index> m_indices;
  std::size_t m_current{};
  std::size_t m_total{};
};

} // namespace detail
} // namespace nvbench
