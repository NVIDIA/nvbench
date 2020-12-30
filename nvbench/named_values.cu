#include <nvbench/named_values.cuh>

#include <fmt/format.h>

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <type_traits>

namespace nvbench
{

void named_values::clear() { m_map.clear(); }

std::size_t named_values::get_size() const { return m_map.size(); }

std::vector<std::string> named_values::get_names() const
{
  std::vector<std::string> names;
  names.reserve(m_map.size());
  std::transform(m_map.cbegin(),
                 m_map.cend(),
                 std::back_inserter(names),
                 [](const auto &val) { return val.first; });
  return names;
}

bool named_values::has_value(const std::string &name) const
{
  return m_map.find(name) != m_map.cend();
}

const named_values::value_type &
named_values::get_value(const std::string &name) const
{
  return m_map.at(name);
}

named_values::type named_values::get_type(const std::string &name) const
{
  return std::visit(
    [&name]([[maybe_unused]] auto &&arg) {
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, nvbench::int64_t>)
      {
        return nvbench::named_values::type::int64;
      }
      else if constexpr (std::is_same_v<T, nvbench::float64_t>)
      {
        return nvbench::named_values::type::float64;
      }
      else if constexpr (std::is_same_v<T, std::string>)
      {
        return nvbench::named_values::type::string;
      }
      throw std::runtime_error(fmt::format("{}:{}: Unknown variant type for "
                                           "entry '{}'.",
                                           __FILE__,
                                           __LINE__,
                                           name));
    },
    this->get_value(name));
}

nvbench::int64_t named_values::get_int64(const std::string &name) const
{
  return std::get<nvbench::int64_t>(this->get_value(name));
}

nvbench::float64_t named_values::get_float64(const std::string &name) const
{
  return std::get<nvbench::float64_t>(this->get_value(name));
}

const std::string &named_values::get_string(const std::string &name) const
{
  return std::get<std::string>(this->get_value(name));
}

void named_values::set_int64(std::string name, nvbench::int64_t value)
{
  m_map.insert(std::make_pair(std::move(name), value_type{std::move(value)}));
}

void named_values::set_float64(std::string name, nvbench::float64_t value)
{
  m_map.insert(std::make_pair(std::move(name), value_type{std::move(value)}));
}

void named_values::set_string(std::string name, std::string value)
{
  m_map.insert(std::make_pair(std::move(name), value_type{std::move(value)}));
}

void named_values::set_value(std::string name, named_values::value_type value)
{
  m_map.insert(std::make_pair(std::move(name), std::move(value)));
}

void named_values::remove_value(const std::string &name)
{
  auto iter = m_map.find(name);
  if (iter != m_map.end())
  {
    m_map.erase(iter);
  }
}

} // namespace nvbench
