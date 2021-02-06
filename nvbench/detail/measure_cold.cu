#include <nvbench/detail/measure_cold.cuh>

#include <nvbench/benchmark_base.cuh>
#include <nvbench/state.cuh>
#include <nvbench/summary.cuh>

#include <fmt/format.h>

#include <algorithm>
#include <cstdio>
#include <variant>

namespace nvbench
{

namespace detail
{

void measure_cold_base::generate_summaries()
{
  {
    auto &summ = m_state.add_summary("Number of Trials (Cold)");
    summ.set_string("short_name", "Cold Trials");
    summ.set_string("description",
                    "Number of kernel executions in cold time measurements.");
    summ.set_int64("value", m_total_iters);
  }

  const auto avg_cuda_time = m_total_cuda_time / m_total_iters;
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
    summ.set_string("short_name", "GPU Noise");
    summ.set_string("description",
                    "Relative standard deviation of the cold GPU execution "
                    "time measurements.");
    summ.set_float64("value", m_cuda_noise);
  }

  const auto avg_cpu_time = m_total_cpu_time / m_total_iters;
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
    summ.set_string("short_name", "CPU Noise");
    summ.set_string("description",
                    "Relative standard deviation of the cold CPU execution "
                    "time measurements.");
    summ.set_float64("value", m_cpu_noise);
  }

  // Log to stdout:
  fmt::memory_buffer param_buffer;
  fmt::format_to(param_buffer, "");
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

  fmt::print("`{}` [{}] Cold {:.6f}ms GPU, {:.6f}ms CPU, {:0.2f}s total, "
             "{}x\n",
             m_state.get_benchmark().get_name(),
             fmt::to_string(param_buffer),
             avg_cuda_time * 1e3,
             avg_cpu_time * 1e3,
             std::max(m_total_cuda_time, m_total_cpu_time),
             m_total_iters);
  std::fflush(stdout);
}

} // namespace detail

} // namespace nvbench
