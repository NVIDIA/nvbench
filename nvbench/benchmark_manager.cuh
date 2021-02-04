#pragma once

#include <nvbench/benchmark_base.cuh>

#include <memory>
#include <vector>

namespace nvbench
{

/**
 * Singleton class that owns reference copies of all known benchmarks.
 */
struct benchmark_manager
{
  using benchmark_vector =
    std::vector<std::unique_ptr<nvbench::benchmark_base>>;

  /**
   * @return The singleton benchmark_manager instance.
   */
  [[nodiscard]] static benchmark_manager &get();

  /**
   * Register a new benchmark.
   */
  benchmark_base &add(std::unique_ptr<benchmark_base> bench);

  /**
   * Clone all benchmarks in the manager into the returned vector.
   */
  [[nodiscard]] benchmark_vector clone_benchmarks() const;

  /**
   * Get a non-mutable reference to benchmark with the specified name.
   */
  [[nodiscard]] const benchmark_base &
  get_benchmark(const std::string &name) const;

  [[nodiscard]] const benchmark_vector &get_benchmarks() const
  {
    return m_benchmarks;
  };

private:
  benchmark_manager()                          = default;
  benchmark_manager(const benchmark_manager &) = delete;
  benchmark_manager(benchmark_manager &&)      = delete;
  benchmark_manager &operator=(const benchmark_manager &) = delete;
  benchmark_manager &operator=(benchmark_manager &&) = delete;

  benchmark_vector m_benchmarks;
};

} // namespace nvbench
