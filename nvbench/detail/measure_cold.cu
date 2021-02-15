#include <nvbench/detail/measure_cold.cuh>

#include <nvbench/benchmark_base.cuh>
#include <nvbench/state.cuh>
#include <nvbench/summary.cuh>

#include <fmt/format.h>

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <variant>

namespace nvbench::detail
{

measure_cold_base::measure_cold_base(state &exec_state)
    : m_state{exec_state}
    , m_min_samples{exec_state.get_min_samples()}
    , m_max_noise{exec_state.get_max_noise()}
    , m_min_time{exec_state.get_min_time()}
    , m_timeout{exec_state.get_timeout()}
{}

void measure_cold_base::check()
{
  const auto device = m_state.get_device();
  if (!device)
  {
    throw std::runtime_error(fmt::format("{}:{}: Device required for `cold` "
                                         "measurement.",
                                         __FILE__,
                                         __LINE__));
  }
  if (!device->is_active())
  { // This means something went wrong higher up. Throw an error.
    throw std::runtime_error(fmt::format("{}:{}: Internal error: Current "
                                         "device is not active.",
                                         __FILE__,
                                         __LINE__));
  }
}

void measure_cold_base::generate_summaries()
{
  const auto d_samples     = static_cast<double>(m_total_samples);
  const auto avg_cuda_time = m_total_cuda_time / d_samples;
  {
    auto &summ = m_state.add_summary("Average GPU Time (Cold)");
    summ.set_string("hint", "duration");
    summ.set_string("short_name", "Cold GPU");
    summ.set_string("description",
                    "Average isolated kernel execution time as measured "
                    "by CUDA events.");
    summ.set_float64("value", avg_cuda_time);
  }

  {
    auto &summ = m_state.add_summary("GPU Relative Standard Deviation (Cold)");
    summ.set_string("hint", "percentage");
    summ.set_string("short_name", "Noise");
    summ.set_string("description",
                    "Relative standard deviation of the cold GPU execution "
                    "time measurements.");
    summ.set_float64("value", m_cuda_noise);
  }

  const auto avg_cpu_time = m_total_cpu_time / d_samples;
  {
    auto &summ = m_state.add_summary("Average CPU Time (Cold)");
    summ.set_string("hint", "duration");
    summ.set_string("short_name", "Cold CPU");
    summ.set_string("description",
                    "Average isolated kernel execution time observed "
                    "from host.");
    summ.set_float64("value", avg_cpu_time);
  }

  {
    auto &summ = m_state.add_summary("CPU Relative Standard Deviation (Cold)");
    summ.set_string("hint", "percentage");
    summ.set_string("short_name", "Noise");
    summ.set_string("description",
                    "Relative standard deviation of the cold CPU execution "
                    "time measurements.");
    summ.set_float64("value", m_cpu_noise);
  }

  {
    auto &summ = m_state.add_summary("Number of Samples (Cold)");
    summ.set_string("short_name", "Samples");
    summ.set_string("description",
                    "Number of kernel executions in cold time measurements.");
    summ.set_int64("value", m_total_samples);
  }

  // Log to stdout:
  fmt::memory_buffer param_buffer;
  fmt::format_to(param_buffer, "Device={}", m_state.get_device()->get_id());
  const axes_metadata &axes = m_state.get_benchmark().get_axes();
  const auto &axis_values   = m_state.get_axis_values();
  for (const auto &name : axis_values.get_names())
  {
    if (param_buffer.size() != 0)
    {
      param_buffer.push_back(' ');
    }
    fmt::format_to(param_buffer, "{}=", name);

    // Handle power-of-two int64 axes differently:
    if (axis_values.get_type(name) == named_values::type::int64 &&
        axes.get_int64_axis(name).is_power_of_two())
    {
      const nvbench::uint64_t value    = axis_values.get_int64(name);
      const nvbench::uint64_t exponent = int64_axis::compute_log2(value);
      fmt::format_to(param_buffer, "2^{}", exponent);
    }
    else
    {
      std::visit(
        [&param_buffer](const auto &val) {
          fmt::format_to(param_buffer, "{}", val);
        },
        axis_values.get_value(name));
    }
  }

  fmt::print("`{}` [{}] Cold {:.6f}ms GPU, {:.6f}ms CPU, {:0.2f}s total GPU, "
             "{}x\n",
             m_state.get_benchmark().get_name(),
             fmt::to_string(param_buffer),
             avg_cuda_time * 1e3,
             avg_cpu_time * 1e3,
             m_total_cuda_time,
             m_total_samples);
  if (m_max_time_exceeded)
  {
    if (m_cuda_noise > m_max_noise)
    {
      fmt::print("!!!! Previous benchmark exceeded max time while over "
                 "noise threshold ({:0.2f}% > {:0.2f}%)\n",
                 m_cuda_noise,
                 m_max_noise);
    }
    if (m_total_samples < m_min_samples)
    {
      fmt::print("!!!! Previous benchmark exceeded max time before "
                 "accumulating min samples ({} < {})\n",
                 m_total_samples,
                 m_min_samples);
    }
    if (m_total_cuda_time < m_min_time)
    {
      fmt::print("!!!! Previous benchmark exceeded max time before "
                 "accumulating min sample time ({:.2f}s < {:.2f}s)\n",
                 m_total_cuda_time,
                 m_min_time);
    }
  }
  std::fflush(stdout);
}

} // namespace detail

} // namespace nvbench::detail
