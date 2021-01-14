#pragma once

#include <nvbench/axis_base.cuh>

#include <nvbench/flags.cuh>
#include <nvbench/types.cuh>

#include <string>
#include <vector>

namespace nvbench
{

enum class int64_axis_flags
{
  none         = 0,
  power_of_two = 0x1
};

} // namespace nvbench

NVBENCH_DECLARE_FLAGS(nvbench::int64_axis_flags);

namespace nvbench
{

struct int64_axis final : public axis_base
{
  int64_axis(std::string name, int64_axis_flags flags = int64_axis_flags::none)
      : axis_base{std::move(name), axis_type::int64}
      , m_inputs{}
      , m_values{}
      , m_flags{flags}
  {}

  ~int64_axis() final;

  [[nodiscard]] bool is_power_of_two() const
  {
    return static_cast<bool>(m_flags & int64_axis_flags::power_of_two);
  }

  void set_inputs(std::vector<int64_t> inputs);

  [[nodiscard]] const std::vector<int64_t> &get_inputs() const
  {
    return m_inputs;
  };

  [[nodiscard]] int64_t get_value(std::size_t i) const
  {
    return m_values[i];
  };

  [[nodiscard]] const std::vector<int64_t> &get_values() const
  {
    return m_values;
  };

  static nvbench::int64_t compute_pow2(nvbench::int64_t exponent)
  {
    return 1ll << exponent;
  }

  static nvbench::int64_t compute_log2(nvbench::int64_t value)
  {
    // TODO use <bit> functions in C++20
    nvbench::uint64_t bits = static_cast<nvbench::uint64_t>(value);
    nvbench::int64_t exponent = 0;
    while (bits >>= 1)
    {
      ++exponent;
    }
    return exponent;
  };

private:
  std::unique_ptr<axis_base> do_clone() const
  {
    return std::make_unique<int64_axis>(*this);
  }
  std::size_t do_get_size() const final { return m_inputs.size(); }
  std::string do_get_input_string(std::size_t) const final;
  std::string do_get_description(std::size_t) const final;

  std::vector<int64_t> m_inputs;
  std::vector<int64_t> m_values;
  int64_axis_flags m_flags;
};

} // namespace nvbench
