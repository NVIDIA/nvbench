#pragma once

#include <nvbench/axis_base.cuh>
#include <nvbench/types.cuh>

#include <string>
#include <vector>

namespace nvbench
{

struct int64_axis final : public axis_base
{
  int64_axis(std::string name, bool is_power_of_two)
    : axis_base{std::move(name), axis_type::int64}
    , m_is_power_of_two{is_power_of_two}
  {}

  ~int64_axis() final;

  [[nodiscard]] bool get_is_power_of_two() const { return m_is_power_of_two; }

  void set_inputs(const std::vector<int64_t> &inputs);

  [[nodiscard]] const std::vector<int64_t> &get_inputs() const
  {
    return m_inputs;
  };

  [[nodiscard]] const std::vector<int64_t> &get_values() const
  {
    return m_values;
  };

  std::size_t do_get_size() const final { return m_inputs.size(); }
  std::string do_get_user_string(std::size_t) const final;
  std::string do_get_user_description(std::size_t) const final;

private:
  std::vector<int64_t> m_inputs;
  std::vector<int64_t> m_values;
  bool m_is_power_of_two;
};

} //
