#pragma once

#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

namespace nvbench
{

struct benchmark_base;
struct state;

enum class log_level
{
  Run,
  Pass,
  Fail,
  Skip,
  Warn,
  Info
};

/*!
 * Base class for all output formats.
 *
 * Simple file writers probably just want to override `print_benchmark_results`
 * and do everything in there.
 *
 * Interactive terminal output formats should implement the logging functions so
 * users have some feedback.
 */
struct output_format
{
  using benchmark_vector = std::vector<std::unique_ptr<benchmark_base>>;

  /*!
   * Construct a new output_format that will write to ostream.
   */
  explicit output_format(std::ostream &ostream);
  ~output_format();

  // move-only
  output_format(const output_format &) = delete;
  output_format(output_format &&)      = default;
  output_format &operator=(const output_format &) = delete;
  output_format &operator=(output_format &&) = default;

  /*!
   * Print a summary of all detected devices, if supported.
   *
   * Called before running benchmarks for active terminal output.
   */
  void print_device_info() { this->do_print_device_info(); }

  /*!
   * Called before/after starting benchmarks and submitting logs.
   * @{
   */
  void print_log_preamble() { this->do_print_log_preamble(); }
  void print_log_epilogue() { this->do_print_log_epilogue(); }
  /*!@}*/

  /*!
   * Print a log message at the specified log level.
   */
  void log(nvbench::log_level level, const std::string &msg)
  {
    this->do_log(level, msg);
  }

  /*!
   * Called before running the measurements associated with state.
   * Implementations are expected to call `log(log_level::Run, ...)`.
   */
  void log_run_state(const nvbench::state &exec_state)
  {
    this->do_log_run_state(exec_state);
  }

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
  // Implementation hooks for subclasses:
  virtual void do_print_device_info() {}
  virtual void do_print_log_preamble() {}
  virtual void do_print_log_epilogue() {}
  virtual void do_log(nvbench::log_level, const std::string &) {}
  virtual void do_log_run_state(const nvbench::state &) {}
  virtual void do_print_benchmark_list(const benchmark_vector &) {}
  virtual void do_print_benchmark_results(const benchmark_vector &benches) {}
};

} // namespace nvbench
