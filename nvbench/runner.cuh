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
      nvbench::detail::state_generator::create(m_benchmark);
  }

  void run()
  {
    if (m_benchmark.m_devices.empty())
    {
      this->run_device(std::nullopt);
    }
    else
    {
      for (const auto &device : m_benchmark.m_devices)
      {
        this->run_device(device);
      }
    }
  }

private:

  void run_device(const std::optional<nvbench::device_info> &device)
  {
    if (device)
    {
      device->set_active();
    }

    // Iterate through type_configs:
    std::size_t type_config_index = 0;
    nvbench::tl::foreach<type_configs>([&states = m_benchmark.m_states,
                                        &type_config_index,
                                        &device](auto type_config_wrapper) {

      // Get current type_config:
      using type_config = typename decltype(type_config_wrapper)::type;

      // Find states with the current device / type_config
      for (nvbench::state &cur_state : states)
      {
        if (cur_state.get_device() == device &&
            cur_state.get_type_config_index() == type_config_index)
        {
          kernel_generator{}(cur_state, type_config{});
        }
      }

      ++type_config_index;
    });
  }

  benchmark_type &m_benchmark;
};

} // namespace nvbench
