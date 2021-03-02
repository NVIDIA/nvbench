#pragma once

#include <nvbench/printer_base.cuh>

namespace nvbench
{

/*!
 * CSV output format.
 */
struct csv_printer : nvbench::printer_base
{
  using printer_base::printer_base;

private:
  // Virtual API from printer_base:
  void do_print_benchmark_results(const benchmark_vector &benches) override;
};

} // namespace nvbench
