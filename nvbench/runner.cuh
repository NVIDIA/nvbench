#pragma once

#include <nvbench/detail/state_generator.cuh>

#include <stdexcept>
#include <vector>

namespace nvbench
{

template <typename BenchmarkType>
struct runner
{
  using benchmark_type   = BenchmarkType;
  using kernel_generator = typename benchmark_type::kernel_generator;
  using type_configs     = typename benchmark_type::type_configs;
  static constexpr std::size_t num_type_configs =
    benchmark_type::num_type_configs;

  explicit runner(benchmark_type &bench)
      : m_benchmark{bench}
  {}

  void generate_states()
  {
    m_benchmark.m_states =
      nvbench::detail::state_generator::create(m_benchmark.m_axes);
  }

  void run()
  {
    auto states_iter = m_benchmark.m_states.begin();
    if (states_iter + num_type_configs != m_benchmark.m_states.end())
    {
      throw std::runtime_error("State vector doesn't match type_configs.");
    }

    nvbench::tl::foreach<type_configs>(
      [&states_iter](auto type_config_wrapper) {
        using type_config = typename decltype(type_config_wrapper)::type;
        for (nvbench::state &cur_state : *states_iter)
        {
          kernel_generator{}(cur_state, type_config{});
        }
        states_iter++;
      });
  }

private:
  benchmark_type &m_benchmark;
};

} // namespace nvbench
