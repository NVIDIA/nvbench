#pragma once

#include <nvbench/output_format.cuh>

namespace nvbench
{

/*!
 * CSV output format.
 */
struct csv_format : nvbench::output_format
{
  using output_format::output_format;

private:
  // Virtual API from output_format:
  void do_print_benchmark_results(const benchmark_vector &benches) override;
};

} // namespace nvbench
