#pragma once

#include <nvbench/types.cuh>

#include <unordered_map>

namespace nvbench
{

struct params
{
  [[nodiscard]] const std::string &
  get_string_param(const std::string &axis_name) const;

  [[nodiscard]] nvbench::int64_t
  get_int64_param(const std::string &axis_name) const;

  [[nodiscard]] nvbench::float64_t
  get_float64_param(const std::string &axis_name) const;

  void add_string_param(std::string axis_name, std::string value);
  void add_int64_param(std::string axis_name, nvbench::int64_t value);
  void add_float64_param(std::string axis_name, nvbench::float64_t value);

private:
  std::unordered_map<std::string, std::string> m_string_params;
  std::unordered_map<std::string, nvbench::int64_t> m_int64_params;
  std::unordered_map<std::string, nvbench::float64_t> m_float64_params;
};

} // namespace nvbench
