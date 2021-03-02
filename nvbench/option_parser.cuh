#pragma once

#include <nvbench/output_multiplex.cuh>

#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

namespace nvbench
{

struct benchmark_base;
struct float64_axis;
struct int64_axis;
struct output_format;
struct string_axis;
struct type_axis;

/**
 * Parses command-line args into a set of benchmarks.
 */
struct option_parser
{
  using benchmark_vector =
    std::vector<std::unique_ptr<nvbench::benchmark_base>>;

  option_parser();
  ~option_parser();

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

  /*!
   * Returns the output format requested by the parse options.
   *
   * If no output format requested, markdown + stdout are used.
   *
   * If multiple formats requested, an output_multiple is used.
   *
   * The returned object is only valid for the lifetime of this option_parser.
   */
  // output_format has no useful const API, so no const overload.
  [[nodiscard]] nvbench::output_format &get_printer();

private:
  void parse_impl();

  using arg_iterator_t = std::vector<std::string>::const_iterator;
  void parse_range(arg_iterator_t first, arg_iterator_t last);

  void add_markdown_format(const std::string &spec);
  void add_csv_format(const std::string &spec);

  std::ostream &output_format_spec_to_ostream(const std::string &spec);

  void print_list() const;

  void add_benchmark(const std::string &name);
  void replay_global_args();

  void update_devices(const std::string &devices);

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

  void update_int64_prop(const std::string &prop_arg,
                         const std::string &prop_val);
  void update_float64_prop(const std::string &prop_arg,
                           const std::string &prop_val);

  // less gross argv:
  std::vector<std::string> m_args;

  // Store benchmark modifiers passed in before any benchmarks are requested as
  // "global args". Replay them after every benchmark.
  std::vector<std::string> m_global_args;
  benchmark_vector m_benchmarks;

  // Manages lifetimes of any ofstreams opened for m_printer.
  std::vector<std::unique_ptr<std::ofstream>> m_ofstream_storage;

  // The main printer to use:
  nvbench::output_multiplex m_printer;

  // Use color on any stdout markdown printers.
  bool m_color_md_stdout_printer{false};

  // True if any stdout printers have been added to m_printer.
  bool m_have_stdout_printer{false};
};

} // namespace nvbench
