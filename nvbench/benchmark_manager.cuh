#pragma once

#include <nvbench/benchmark_base.cuh>

#include <memory>
#include <vector>

namespace nvbench
{

/**
 * Singleton class that owns all benchmarks.
 */
struct benchmark_manager
{
  using benchmark_vector =
    std::vector<std::unique_ptr<nvbench::benchmark_base>>;

  [[nodiscard]] static benchmark_manager &get();

  benchmark_base &add(std::unique_ptr<benchmark_base> bench);

  [[nodiscard]] benchmark_base &get_benchmark(const std::string &name);
  [[nodiscard]] const benchmark_base &
  get_benchmark(const std::string &name) const;

  [[nodiscard]] benchmark_vector &get_benchmarks() { return m_benchmarks; };
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
