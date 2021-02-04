#include <nvbench/benchmark_manager.cuh>

#include <fmt/format.h>

#include <algorithm>
#include <stdexcept>

namespace nvbench
{

benchmark_manager &benchmark_manager::get()
{ // Karen's function:
  static benchmark_manager the_manager;
  return the_manager;
}

benchmark_base &benchmark_manager::add(std::unique_ptr<benchmark_base> bench)
{
  m_benchmarks.push_back(std::move(bench));
  return *m_benchmarks.back();
}

benchmark_manager::benchmark_vector benchmark_manager::clone_benchmarks() const
{
  benchmark_vector result(m_benchmarks.size());
  std::transform(m_benchmarks.cbegin(),
                 m_benchmarks.cend(),
                 result.begin(),
                 [](const auto &bench) { return bench->clone(); });
  return result;
}

const benchmark_base &
benchmark_manager::get_benchmark(const std::string &name) const
{
  auto iter = std::find_if(m_benchmarks.cbegin(),
                           m_benchmarks.cend(),
                           [&name](const auto &bench_ptr) {
                             return bench_ptr->get_name() == name;
                           });
  if (iter == m_benchmarks.cend())
  {
    throw std::runtime_error(
      fmt::format("{}:{}: No benchmark named '{}'.", name));
  }

  return **iter;
}

} // namespace nvbench
