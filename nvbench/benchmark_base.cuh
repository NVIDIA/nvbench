#pragma once

#include <nvbench/axes_metadata.cuh>

namespace nvbench
{

/**
 * Hold runtime benchmark information and provides public customization API for
 * the `NVBENCH_CREATE` macros.
 *
 * Delegates responsibility to the following classes:
 * - nvbench::axes_metadata: Axis specifications.
 */
struct benchmark_base
{
  virtual ~benchmark_base();

  benchmark_base &set_name(std::string name)
  {
    m_name = std::move(name);
    return *this;
  }

  [[nodiscard]] const std::string &get_name() const { return m_name; }

  // Convenience API for a single type_axis.
  benchmark_base &set_type_axes_name(std::string name)
  {
    return this->set_type_axes_names({std::move(name)});
  }

  benchmark_base &set_type_axes_names(std::vector<std::string> names)
  {
    this->do_set_type_axes_names(std::move(names));
    return *this;
  }

  benchmark_base &add_float64_axis(std::string name,
                                   std::vector<nvbench::float64_t> data)
  {
    m_axes.add_float64_axis(std::move(name), std::move(data));
    return *this;
  }

  benchmark_base &add_int64_axis(
    std::string name,
    std::vector<nvbench::int64_t> data,
    nvbench::int64_axis_flags flags = nvbench::int64_axis_flags::none)
  {
    m_axes.add_int64_axis(std::move(name), std::move(data), flags);
    return *this;
  }

  benchmark_base &add_int64_power_of_two_axis(std::string name,
                                              std::vector<nvbench::int64_t> data)
  {
    return this->add_int64_axis(std::move(name),
                                std::move(data),
                                nvbench::int64_axis_flags::power_of_two);
  }

  benchmark_base &add_string_axis(std::string name,
                                  std::vector<std::string> data)
  {
    m_axes.add_string_axis(std::move(name), std::move(data));
    return *this;
  }

  [[nodiscard]] const nvbench::axes_metadata &get_axes() const
  {
    return m_axes;
  }

protected:
  std::string m_name;
  nvbench::axes_metadata m_axes;

private:
  // route this through a virtual so the templated subclass can inject type info
  virtual void do_set_type_axes_names(std::vector<std::string> names) = 0;
};

} // namespace nvbench
