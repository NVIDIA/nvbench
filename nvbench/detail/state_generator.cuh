#pragma once

#include <nvbench/axes_metadata.cuh>
#include <nvbench/axis_base.cuh>
#include <nvbench/state.cuh>

#include <string>
#include <vector>

namespace nvbench
{

namespace detail
{

struct state_generator
{

  static std::vector<std::vector<nvbench::state>>
  create(const axes_metadata &axes);

protected:
  struct axis_index
  {
    std::string axis;
    nvbench::axis_type type;
    std::size_t index;
    std::size_t size;
  };

  void add_axis(const nvbench::axis_base &axis)
  {
    this->add_axis(axis.get_name(), axis.get_type(), axis.get_size());
  }

  void add_axis(std::string axis, nvbench::axis_type type, std::size_t size)
  {
    m_indices.push_back({std::move(axis), type, std::size_t{0}, size});
  }

  [[nodiscard]] std::size_t get_number_of_states() const;

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

  std::vector<axis_index> m_indices;
  std::size_t m_current{};
  std::size_t m_total{};
};

} // namespace detail
} // namespace nvbench
