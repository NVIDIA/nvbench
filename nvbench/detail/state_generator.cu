#include <nvbench/detail/state_generator.cuh>

#include <functional>
#include <numeric>

namespace nvbench
{

namespace detail
{

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
  m_total = this->get_number_of_states();
  for (axis_index &entry : m_indices)
  {
    entry.index = 0;
  }
}

bool state_generator::iter_valid() const { return m_current < m_total; }

void state_generator::next()
{
  for (axis_index& axis_info : m_indices)
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
