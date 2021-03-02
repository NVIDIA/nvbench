#pragma once

#include <nvbench/printer_base.cuh>

#include <memory>
#include <vector>

namespace nvbench
{

/*!
 * An nvbench::printer_base that just forwards calls to other `printer_base`s.
 */
struct output_multiplex : nvbench::printer_base
{

  output_multiplex();

  template <typename Format, typename... Ts>
  Format &emplace(Ts &&...ts)
  {
    m_printers.push_back(std::make_unique<Format>(std::forward<Ts>(ts)...));
    return static_cast<Format &>(*m_printers.back());
  }

  [[nodiscard]] std::size_t get_output_count() const
  {
    return m_printers.size();
  }

private:
  void do_print_device_info() override;
  void do_print_log_preamble() override;
  void do_print_log_epilogue() override;
  void do_log(nvbench::log_level, const std::string &) override;
  void do_log_run_state(const nvbench::state &) override;
  void do_print_benchmark_list(const benchmark_vector &benches) override;
  void do_print_benchmark_results(const benchmark_vector &benches) override;

  std::vector<std::unique_ptr<nvbench::printer_base>> m_printers;
};

} // namespace nvbench
