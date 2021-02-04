#pragma once

#include <nvbench/benchmark_base.cuh>

#include <memory>
#include <vector>

namespace nvbench
{

namespace detail
{

struct markdown_format
{
  using benchmark_vector = std::vector<std::unique_ptr<benchmark_base>>;

  // Hacked in to just print a basic summary table to stdout. There's lots of
  // room for improvement here.
  static void print(const benchmark_vector &benchmarks);
};

} // namespace detail
} // namespace nvbench
