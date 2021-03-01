#pragma once

#include <iosfwd>
#include <memory>
#include <vector>

namespace nvbench
{

struct benchmark_base;

/*!
 * Base class for all output formats.
 *
 * Simple file writers probably just want to override `print_benchmark_results`
 * and do everything in there.
 *
 * Interactive terminal output formats should implement the logging functions.
 */
struct output_format
{
  using benchmark_vector = std::vector<std::unique_ptr<benchmark_base>>;

  /*!
   * Construct a new output_format that will write to ostream.
   */
  explicit output_format(std::ostream &ostream);
  ~output_format();

  /*!
   * Print a summary of all detected devices, if supported.
   *
   * Called before running benchmarks for active terminal output.
   */
  void print_device_info() { this->do_print_device_info(); }

  /*!
   * Called before/after logging for active terminal output.
   * @{
   */
  void print_log_preamble() { this->do_print_log_preamble(); }
  void print_log_epilogue() { this->do_print_log_epilogue(); }
  /*!@}*/

  /*!
   * Print details of the unexecuted benchmarks in `benches`. This is used for
   * `--list`.
   */
  void print_benchmark_list(const benchmark_vector &benches)
  {
    this->do_print_benchmark_list(benches);
  }

  /*!
   * Print the results of the benchmarks in `benches`.
   */
  void print_benchmark_results(const benchmark_vector &benches)
  {
    this->do_print_benchmark_results(benches);
  }

protected:
  std::ostream &m_ostream;

private:
  virtual void do_print_device_info() {}
  virtual void do_print_log_preamble() {}
  virtual void do_print_log_epilogue() {}
  virtual void do_print_benchmark_list(const benchmark_vector &){};
  virtual void do_print_benchmark_results(const benchmark_vector &benches) = 0;
};

} // namespace nvbench
