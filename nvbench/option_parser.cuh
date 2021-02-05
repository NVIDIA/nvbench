#pragma once

#include <nvbench/benchmark_base.cuh>

#include <memory>
#include <string>
#include <vector>

namespace nvbench
{

/**
 * Parses command-line args into a set of benchmarks.
 */
struct option_parser
{
  using benchmark_vector =
    std::vector<std::unique_ptr<nvbench::benchmark_base>>;

  void parse(int argc, char const *const argv[]);
  void parse(std::vector<std::string> args);

  [[nodiscard]] benchmark_vector &get_benchmarks() { return m_benchmarks; };
  [[nodiscard]] const benchmark_vector &get_benchmarks() const
  {
    return m_benchmarks;
  };

  [[nodiscard]] const std::vector<std::string> &get_args() const
  {
    return m_args;
  }

private:
  void parse_impl();
  void print_list() const;
  void add_benchmark(const std::string &name);
  void update_axis(const std::string &spec);
  static void update_int64_axis(int64_axis &axis,
                                std::string_view value_spec,
                                std::string_view flag_spec);
  static void update_float64_axis(float64_axis &axis,
                                  std::string_view value_spec,
                                  std::string_view flag_spec);
  static void update_string_axis(string_axis &axis,
                                 std::string_view value_spec,
                                 std::string_view flag_spec);
  static void update_type_axis(type_axis &axis,
                               std::string_view value_spec,
                               std::string_view flag_spec);

  std::vector<std::string> m_args;
  benchmark_vector m_benchmarks;
};

} // namespace nvbench
