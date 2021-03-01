#pragma once

#include <nvbench/output_format.cuh>

#include <string>

namespace nvbench
{

struct summary;

/*!
 * Markdown output format.
 *
 * Includes customization points to modify numeric formatting.
 */
struct markdown_format : nvbench::output_format
{
  using output_format::output_format;

private:
  // Virtual API from output_format:
  void do_print_device_info() override;
  void do_print_log_preamble() override;
  void do_print_log_epilogue() override;
  void do_print_benchmark_list(const benchmark_vector &benches) override;
  void do_print_benchmark_results(const benchmark_vector &benches) override;

  // Customization points for formatting:
  virtual std::string do_format_default(const nvbench::summary &data);
  virtual std::string do_format_duration(const nvbench::summary &seconds);
  virtual std::string do_format_item_rate(const nvbench::summary &items_per_sec);
  virtual std::string do_format_bytes(const nvbench::summary &bytes);
  virtual std::string do_format_byte_rate(const nvbench::summary &bytes_per_sec);
  virtual std::string do_format_sample_size(const nvbench::summary &count);
  virtual std::string do_format_percentage(const nvbench::summary &percentage);
};

} // namespace nvbench
