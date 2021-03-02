#include <nvbench/runner.cuh>

#include <nvbench/benchmark_base.cuh>
#include <nvbench/output_format.cuh>
#include <nvbench/state.cuh>

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
  // the error already.
  if (exec_state.is_skipped())
  {
    this->print_skip_notification(exec_state);
  }
  else
  {
    const auto reason = fmt::format("Unexpected error: {}", e.what());

    if (auto printer_opt_ref = exec_state.get_benchmark().get_printer();
        printer_opt_ref.has_value())
    {
      auto &printer = printer_opt_ref.value().get();
      printer.log(nvbench::log_level::Fail, reason);
    }

    exec_state.skip(reason);
  }
}

void runner_base::announce_state(nvbench::state &exec_state) const
{
  // Log if a printer exists:
  if (auto printer_opt_ref = exec_state.get_benchmark().get_printer();
      printer_opt_ref.has_value())
  {
    auto &printer = printer_opt_ref.value().get();
    printer.log_run_state(exec_state);
  }
}

void runner_base::print_skip_notification(state &exec_state) const
{
  if (auto printer_opt_ref = exec_state.get_benchmark().get_printer();
      printer_opt_ref.has_value())
  {
    auto &printer = printer_opt_ref.value().get();
    printer.log(nvbench::log_level::Skip, exec_state.get_skip_reason());
  }
}

} // namespace nvbench
