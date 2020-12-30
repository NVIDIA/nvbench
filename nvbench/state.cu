#include <nvbench/state.cuh>

#include <nvbench/types.cuh>

#include <string>

namespace nvbench
{

nvbench::int64_t state::get_int64(const std::string &axis_name) const
{
  return m_axis_values.get_int64(axis_name);
}

nvbench::float64_t state::get_float64(const std::string &axis_name) const
{
  return m_axis_values.get_float64(axis_name);
}

const std::string &state::get_string(const std::string &axis_name) const
{
  return m_axis_values.get_string(axis_name);
}

} // namespace nvbench
