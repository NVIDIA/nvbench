#include <nvbench/runner.cuh>

#include <nvbench/benchmark_base.cuh>
#include <nvbench/state.cuh>

#include <fmt/color.h>
#include <fmt/format.h>

#include <cstdio>
#include <stdexcept>

namespace nvbench
{

void runner_base::generate_states()
{
  m_benchmark.m_states = nvbench::detail::state_generator::create(m_benchmark);
}

void runner_base::handle_sampling_exception(const std::exception &e,
                                            state &exec_state) const
{
  // If the state is skipped, that means the execution framework class handled
  // the error already. Just print the exception text as an info log:
  if (exec_state.is_skipped())
  {
    this->print_skip_notification(exec_state);
  }
  else
  {
    const auto reason = fmt::format("Unexpected error: {}", e.what());

    // An unhandled error occurred. Turn up the volume a bit and mark the
    // state as skipped.
    fmt::print("{} {}\n",
               fmt::format(bg(fmt::color::black) | fg(fmt::color::red) |
                 fmt::emphasis::bold, "{:5}", "Fail:"),
               reason);
    exec_state.skip(reason);
  }
}

void runner_base::announce_state(nvbench::state &exec_state) const
{
  fmt::print("{} {}\n",
             fmt::format(bg(fmt::color::black) | fg(fmt::color::white) |
                           fmt::emphasis::bold,
                         "{:5}",
                         "Run:"),
             exec_state.get_short_description());
  std::fflush(stdout);
}

void runner_base::print_skip_notification(state &exec_state) const
{
  fmt::print("{} {}\n",
             fmt::format(bg(fmt::color::black) | fg(fmt::color::steel_blue) |
                           fmt::emphasis::bold,
                         "{:5}",
                         "Skip:"),
             exec_state.get_skip_reason());
}

} // namespace nvbench
