#include <nvbench/detail/measure_hot.cuh>

#include <nvbench/benchmark_base.cuh>
#include <nvbench/device_info.cuh>
#include <nvbench/state.cuh>
#include <nvbench/summary.cuh>

#include <fmt/color.h>
#include <fmt/format.h>

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <variant>

namespace nvbench::detail
{

measure_hot_base::measure_hot_base(state &exec_state)
    : m_state{exec_state}
    , m_min_samples{exec_state.get_min_samples()}
    , m_min_time{exec_state.get_min_time()}
    , m_skip_time{exec_state.get_skip_time()}
    , m_timeout{exec_state.get_timeout()}
{
  // Since cold measures converge to a stable result, increase the min_samples
  // to match the cold result if available.
  try
  {
    nvbench::int64_t cold_samples =
      m_state.get_summary("Number of Samples (Cold)").get_int64("value");
    m_min_samples = std::max(m_min_samples, cold_samples);

    // If the cold measurement ran successfully, disable skip_time. It'd just
    // be annoying to skip now.
    m_skip_time = -1;
  }
  catch (...)
  {
    // If the above threw an exception, we don't have a cold measurement to use.
    // Estimate a target_time between m_min_time and m_timeout.
    // Use the average of the min_time and timeout, but don't go over 5x
    // min_time in case timeout is huge.
    // We could expose a `target_time` property on benchmark_base/state if
    // needed.
    m_min_time = std::min((m_min_time + m_timeout) / 2., m_min_time * 5);
  }
}

void measure_hot_base::check()
{
  const auto device = m_state.get_device();
  if (!device)
  {
    throw std::runtime_error(fmt::format("{}:{}: Device required for `hot` "
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

void measure_hot_base::generate_summaries()
{
  const auto d_samples     = static_cast<double>(m_total_samples);
  const auto avg_cuda_time = m_total_cuda_time / d_samples;
  {
    auto &summ = m_state.add_summary("Average GPU Time (Hot)");
    summ.set_string("hint", "duration");
    summ.set_string("short_name", "Batch GPU");
    summ.set_string("description",
                    "Average back-to-back kernel execution time as measured "
                    "by CUDA events.");
    summ.set_float64("value", avg_cuda_time);
  }

  {
    auto &summ = m_state.add_summary("Number of Samples (Hot)");
    summ.set_string("hint", "sample_size");
    summ.set_string("short_name", "Batch");
    summ.set_string("description",
                    "Number of kernel executions in hot time measurements.");
    summ.set_int64("value", m_total_samples);
  }

  const auto num_format = fmt::emphasis::bold;
  const auto log_format = bg(fmt::color::black) | fg(fmt::color::dark_green) |
                          fmt::emphasis::bold;
  const auto warning_format = // bold yellow-on-black
    bg(fmt::color::black) | fg(fmt::rgb{160, 135, 0}) | fmt::emphasis::bold;

  // Warn if timed out:
  if (m_max_time_exceeded)
  {
    const auto timeout = m_timeout_timer.get_duration();

    if (m_total_samples < m_min_samples)
    {
      fmt::print("{} {}\n",
                 fmt::format(warning_format, "{:5}", "Warn:"),
                 fmt::format("Current measurement timed out ({}) before "
                             "accumulating min_samples ({} < {})",
                             fmt::format(num_format, "{:.2f}s", timeout),
                             fmt::format(num_format, "{}", m_total_samples),
                             fmt::format(num_format, "{}", m_min_samples)));
    }
    if (m_total_cuda_time < m_min_time)
    {
      fmt::print(
        "{} {}\n",
        fmt::format(warning_format, "{:5}", "Warn:"),
        fmt::format("Current measurement timed out ({}) before accumulating "
                    "min_time ({} < {})",
                    fmt::format(num_format, "{:.2f}s", timeout),
                    fmt::format(num_format, "{:0.2}s", m_total_cuda_time),
                    fmt::format(num_format, "{:0.2}s", m_min_time)));
    }
  }

  // Log to stdout:
  fmt::print(
    "{} {}\n",
    fmt::format(log_format, "{:5}", "Hot:"),
    fmt::format("{} GPU, {} total GPU, {}",
                fmt::format(num_format, "{:0.6f}ms", avg_cuda_time * 1e3),
                fmt::format(num_format, "{:.2f}s", m_total_cuda_time),
                fmt::format(num_format, "{}x", m_total_samples)));

  std::fflush(stdout);
}

void measure_hot_base::check_skip_time(nvbench::float64_t warmup_time)
{
  if (m_skip_time > 0. && warmup_time < m_skip_time)
  {
    auto reason = fmt::format(
      "Warmup time did not meet skip_time limit: "
      "{} < {}.",
      fmt::format(fmt::emphasis::bold, "{:0.3f} us", warmup_time * 1e6),
      fmt::format(fmt::emphasis::bold, "{:0.3f} us", m_skip_time * 1e6));

    m_state.skip(reason);
    throw std::runtime_error{std::move(reason)};
  }
}

} // namespace nvbench::detail
