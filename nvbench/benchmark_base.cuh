#pragma once

#include <nvbench/axes_metadata.cuh>
#include <nvbench/state.cuh>

#include <memory>
#include <vector>

namespace nvbench
{

template <typename BenchmarkType>
struct runner;

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

  /**
   * Returns a pointer to a new instance of the concrete benchmark<...>
   * subclass.
   *
   * The result will have the same name and axes as the source benchmark.
   * The `get_states()` vector of the result will always be empty.
   */
  [[nodiscard]] std::unique_ptr<benchmark_base> clone() const;

  benchmark_base &set_name(std::string name)
  {
    m_name = std::move(name);
    return *this;
  }

  [[nodiscard]] const std::string &get_name() const { return m_name; }

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

  [[nodiscard]] nvbench::axes_metadata &get_axes()
  {
    return m_axes;
  }

  [[nodiscard]] const nvbench::axes_metadata &get_axes() const
  {
    return m_axes;
  }

  [[nodiscard]] const std::vector<std::vector<nvbench::state>> &
  get_states() const
  {
    return m_states;
  }
  [[nodiscard]] std::vector<std::vector<nvbench::state>> &get_states()
  {
    return m_states;
  }

  void run() { this->do_run(); }

protected:
  template <typename BenchmarkType>
  friend struct runner;

  std::string m_name;
  nvbench::axes_metadata m_axes;
  std::vector<std::vector<nvbench::state>> m_states;

private:
  // route these through virtuals so the templated subclass can inject type info
  virtual std::unique_ptr<benchmark_base> do_clone() const            = 0;
  virtual void do_set_type_axes_names(std::vector<std::string> names) = 0;
  virtual void do_run()                                               = 0;
};

} // namespace nvbench
