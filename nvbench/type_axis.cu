#include <nvbench/type_axis.cuh>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <stdexcept>

namespace nvbench
{

type_axis::~type_axis() = default;

std::size_t type_axis::get_type_index(const std::string &input_string) const
{
  auto it =
    std::find(m_input_strings.cbegin(), m_input_strings.cend(), input_string);
  if (it == m_input_strings.end())
  {
    throw std::runtime_error(
      fmt::format("{}:{}: Invalid input string '{}' for type_axis `{}`.\n"
                  "Valid input strings: {}",
      __FILE__, __LINE__, input_string, this->get_name(), m_input_strings));
  }

  return it - m_input_strings.cbegin();
}

} // namespace nvbench
