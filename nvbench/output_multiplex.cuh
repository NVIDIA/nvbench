#pragma once

#include <nvbench/output_format.cuh>

#include <memory>
#include <vector>

namespace nvbench
{

/*!
 * An nvbench::output_format that just forwards calls to other `output_format`s.
 */
struct output_multiplex : nvbench::output_format
{

  output_multiplex();

  template <typename Format, typename... Ts>
  Format &emplace(Ts &&...ts)
  {
    m_formats.push_back(std::make_unique<Format>(std::forward<Ts>(ts)...));
    return static_cast<Format &>(*m_formats.back());
  }

  [[nodiscard]] std::size_t get_output_count() const
  {
    return m_formats.size();
  }

private:
  void do_print_device_info() override;
  void do_print_log_preamble() override;
  void do_print_log_epilogue() override;
  void do_log(nvbench::log_level, const std::string &) override;
  void do_log_run_state(const nvbench::state &) override;
  void do_print_benchmark_list(const benchmark_vector &benches) override;
  void do_print_benchmark_results(const benchmark_vector &benches) override;

  std::vector<std::unique_ptr<nvbench::output_format>> m_formats;
};

} // namespace nvbench
