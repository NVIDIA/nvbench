#include <nvbench/state.cuh>

#include <nvbench/types.cuh>

#include <string>
#include <unordered_map>
#include <variant>

namespace nvbench
{

const state::param_type &state::get_param(const std::string &axis_name) const
{
  return m_params.at(axis_name);
}

nvbench::int64_t state::get_int64(const std::string &axis_name) const
{
  return std::get<nvbench::int64_t>(m_params.at(axis_name));
}

nvbench::float64_t state::get_float64(const std::string &axis_name) const
{
  return std::get<nvbench::float64_t>(m_params.at(axis_name));
}

const std::string &state::get_string(const std::string &axis_name) const
{
  return std::get<std::string>(m_params.at(axis_name));
}
void state::set_param(std::string axis_name, nvbench::int64_t value)
{
  m_params.insert(std::make_pair(std::move(axis_name), value));
}

void state::set_param(std::string axis_name, nvbench::float64_t value)
{
  m_params.insert(std::make_pair(std::move(axis_name), value));
}

void state::set_param(std::string axis_name, std::string value)
{
  m_params.insert(std::make_pair(std::move(axis_name), std::move(value)));
}

} // namespace nvbench
