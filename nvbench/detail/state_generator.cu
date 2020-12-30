#include <nvbench/detail/state_generator.cuh>

#include <algorithm>
#include <functional>
#include <numeric>

#include <cassert>

namespace nvbench
{

namespace detail
{

std::vector<nvbench::state> state_generator::create(const axes_metadata &axes)
{
  state_generator sg;
  {
    const auto &axes_vec = axes.get_axes();
    std::for_each(axes_vec.cbegin(), axes_vec.cend(), [&sg](const auto &axis) {
      if (axis->get_type() != nvbench::axis_type::type)
      {
        sg.add_axis(*axis);
      }
    });
  }

  std::vector<nvbench::state> states;
  {
    states.reserve(sg.get_number_of_states());
    for (sg.init(); sg.iter_valid(); sg.next())
    {
      nvbench::state state;
      for (const axis_index &axis_info : sg.get_current_indices())
      {
        switch (axis_info.type)
        {
          default:
          case axis_type::type:
            assert("unreachable." && false);
            break;

          case axis_type::int64:
            state.m_axis_values.set_int64(
              axis_info.axis,
              axes.get_int64_axis(axis_info.axis).get_value(axis_info.index));
            break;

          case axis_type::float64:
            state.m_axis_values.set_float64(
              axis_info.axis,
              axes.get_float64_axis(axis_info.axis).get_value(axis_info.index));
            break;

          case axis_type::string:
            state.m_axis_values.set_string(
              axis_info.axis,
              axes.get_string_axis(axis_info.axis).get_value(axis_info.index));
            break;
        }
      }
      states.push_back(std::move(state));
    }
  }

  return states;
}

std::size_t state_generator::get_number_of_states() const
{
  return std::transform_reduce(m_indices.cbegin(),
                               m_indices.cend(),
                               std::size_t{1},
                               std::multiplies<>{},
                               [](const axis_index &size_info) {
                                 return size_info.size;
                               });
}

void state_generator::init()
{
  m_current = 0;
  m_total   = this->get_number_of_states();
  for (axis_index &entry : m_indices)
  {
    entry.index = 0;
  }
}

bool state_generator::iter_valid() const { return m_current < m_total; }

void state_generator::next()
{
  for (axis_index &axis_info : m_indices)
  {
    axis_info.index += 1;
    if (axis_info.index >= axis_info.size)
    {
      axis_info.index = 0;
      continue; // carry the addition to the next entry in m_indices
    }
    break; // done
  }
  m_current += 1;
}

} // namespace detail
} // namespace nvbench
