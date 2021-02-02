#include <nvbench/type_axis.cuh>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <stdexcept>

namespace nvbench
{

type_axis::~type_axis() = default;

void type_axis::set_active_inputs(const std::vector<std::string> &inputs)
{
  m_mask.clear();
  m_mask.resize(m_input_strings.size(), false);
  for (const auto& input : inputs)
  {
    const auto idx = this->get_type_index(input);
    m_mask[idx] = true;
  }
}

bool type_axis::get_is_active(const std::string &input) const
{
  return this->get_is_active(this->get_type_index(input));
}

bool type_axis::get_is_active(std::size_t idx) const
{
  return m_mask.at(idx);
}

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
