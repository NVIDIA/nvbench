#include <nvbench/float64_axis.cuh>

#include <fmt/format.h>

namespace nvbench
{

float64_axis::~float64_axis() = default;

std::string float64_axis::do_get_input_string(std::size_t i) const
{
  return fmt::to_string(m_values[i]);
}

std::string float64_axis::do_get_description(std::size_t i) const { return {}; }

} // namespace nvbench
