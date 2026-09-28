/*
 *  Copyright 2021 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 with the LLVM exception
 *  (the "License"); you may not use this file except in compliance with
 *  the License.
 *
 *  You may obtain a copy of the License at
 *
 *      http://llvm.org/foundation/relicensing/LICENSE.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <nvbench/benchmark_base.cuh>
#include <nvbench/printer_base.cuh>
#include <nvbench/runner.cuh>
#include <nvbench/state.cuh>

#include <fmt/format.h>

#include <algorithm>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string_view>

namespace nvbench
{

void runner_base::generate_states()
{
  m_benchmark.m_states = nvbench::detail::state_generator::create(m_benchmark);
}

void runner_base::handle_sampling_exception(const std::exception &e, state &exec_state) const
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

    if (auto printer_ptr = exec_state.get_benchmark().get_printer())
    {
      auto &printer = *printer_ptr;
      printer.log(nvbench::log_level::fail, reason);
    }

    exec_state.skip(reason);
  }
}

void runner_base::run_state_prologue(nvbench::state &exec_state) const
{
  // Log if a printer exists:
  if (auto printer_ptr = exec_state.get_benchmark().get_printer())
  {
    auto &printer = *printer_ptr;
    printer.log_run_state(exec_state);
  }
}

void runner_base::generate_throughput_summaries(state &exec_state) const
{
  const auto find_summary = [&exec_state](std::string_view tag) -> const nvbench::summary * {
    const auto &summaries = exec_state.get_summaries();
    const auto iter = std::find_if(summaries.cbegin(), summaries.cend(), [tag](const auto &summary) {
      return summary.get_tag() == tag;
    });
    return iter == summaries.cend() ? nullptr : &*iter;
  };

  const auto add_summaries = [&exec_state](nvbench::float64_t mean_time,
                                           std::string_view prefix,
                                           bool add_utilization) {
    if (mean_time <= 0.)
    {
      return;
    }

    if (const auto items = exec_state.get_element_count(); items != 0)
    {
      auto &summ = exec_state.add_summary(fmt::format("{}/bw/item_rate", prefix));
      summ.set_string("name", "Elem/s");
      summ.set_string("hint", "item_rate");
      summ.set_string("description", "Number of input elements processed per second");
      summ.set_float64("value", static_cast<double>(items) / mean_time);
    }

    if (const auto bytes = exec_state.get_global_memory_rw_bytes(); bytes != 0)
    {
      const auto avg_used_gmem_bw = static_cast<double>(bytes) / mean_time;
      {
        auto &summ = exec_state.add_summary(fmt::format("{}/bw/global/bytes_per_second", prefix));
        summ.set_string("name", "GlobalMem BW");
        summ.set_string("hint", "byte_rate");
        summ.set_string("description",
                        add_utilization
                          ? "Number of bytes read/written per second to the CUDA device's global memory"
                          : "Number of bytes read/written per second.");
        summ.set_float64("value", avg_used_gmem_bw);
      }

      if (add_utilization)
      {
        const auto &device = exec_state.get_device();
        if (device)
        {
          const auto peak_gmem_bw =
            static_cast<double>(device->get_global_memory_bus_bandwidth());
          if (peak_gmem_bw > 0.)
          {
            auto &summ = exec_state.add_summary(fmt::format("{}/bw/global/utilization", prefix));
            summ.set_string("name", "BWUtil");
            summ.set_string("hint", "percentage");
            summ.set_string("description",
                            "Global device memory utilization as a percentage of the "
                            "device's peak bandwidth");
            summ.set_float64("value", avg_used_gmem_bw / peak_gmem_bw);
          }
        }
      }
    }
  };

  if (const auto *mean = find_summary("nv/cold/time/gpu/mean"))
  {
    add_summaries(mean->get_float64("value"), "nv/cold", true);
  }
  if (const auto *mean = find_summary("nv/cpu_only/time/cpu/mean"))
  {
    add_summaries(mean->get_float64("value"), "nv/cpu_only", false);
  }
}

void runner_base::run_state_epilogue(state &exec_state) const
{
  // Throughput depends on metadata that benchmark generators may declare after
  // state.exec() returns, so compute it only after the full generator finishes.
  this->generate_throughput_summaries(exec_state);

  // Release per-state stream resources after state execution has completed.
  // See: https://github.com/NVIDIA/nvbench/issues/437
  exec_state.reset_cuda_stream();

  // Notify the printer that the state has completed::
  if (auto printer_ptr = exec_state.get_benchmark().get_printer())
  {
    auto &printer = *printer_ptr;
    printer.add_completed_state();
  }
}

void runner_base::print_skip_notification(state &exec_state) const
{
  if (auto printer_ptr = exec_state.get_benchmark().get_printer())
  {
    auto &printer = *printer_ptr;
    printer.log(nvbench::log_level::skip, exec_state.get_skip_reason());
  }
}

} // namespace nvbench
