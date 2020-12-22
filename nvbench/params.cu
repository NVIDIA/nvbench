#include <nvbench/params.cuh>

#include <unordered_map>
#include <utility>

namespace nvbench
{

const std::string &params::get_string_param(const std::string &axis_name) const
{
  return m_string_params.at(axis_name);
}

nvbench::int64_t params::get_int64_param(const std::string &axis_name) const
{
  return m_int64_params.at(axis_name);
}

nvbench::float64_t params::get_float64_param(const std::string &axis_name) const
{
  return m_float64_params.at(axis_name);
}

void params::add_string_param(std::string axis_name, std::string value)
{
  m_string_params.insert(
    std::make_pair(std::move(axis_name), std::move(value)));
}

void params::add_int64_param(std::string axis_name, nvbench::int64_t value)
{
  m_int64_params.insert(std::make_pair(std::move(axis_name), value));
}

void params::add_float64_param(std::string axis_name, nvbench::float64_t value)
{
  m_float64_params.insert(std::make_pair(std::move(axis_name), value));
}

} // namespace nvbench
