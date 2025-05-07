/*
 *  Copyright 2021-2025 NVIDIA Corporation
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
#include <nvbench/criterion_manager.cuh>
#include <nvbench/detail/measure_cpu_only.cuh>
#include <nvbench/detail/throw.cuh>
#include <nvbench/printer_base.cuh>
#include <nvbench/state.cuh>
#include <nvbench/summary.cuh>

#include <fmt/format.h>

#include <algorithm>
#include <limits>

namespace nvbench::detail
{

measure_cpu_only_base::measure_cpu_only_base(state &exec_state)
    : m_state{exec_state}
    , m_launch(m_state.get_cuda_stream())
    , m_criterion_params{exec_state.get_criterion_params()}
    , m_stopping_criterion{nvbench::criterion_manager::get().get_criterion(
        exec_state.get_stopping_criterion())}
    , m_run_once{exec_state.get_run_once()}
    , m_min_samples{exec_state.get_min_samples()}
    , m_skip_time{exec_state.get_skip_time()}
    , m_timeout{exec_state.get_timeout()}
{
  if (m_min_samples > 0)
  {
    m_cpu_times.reserve(static_cast<std::size_t>(m_min_samples));
  }
}

void measure_cpu_only_base::check()
{
  // no-op
}

void measure_cpu_only_base::initialize()
{

  m_min_cpu_time      = std::numeric_limits<nvbench::float64_t>::max();
  m_max_cpu_time      = std::numeric_limits<nvbench::float64_t>::lowest();
  m_total_cpu_time    = 0.;
  m_total_samples     = 0;
  m_max_time_exceeded = false;

  m_cpu_times.clear();

  m_stopping_criterion.initialize(m_criterion_params);
}

void measure_cpu_only_base::run_trials_prologue() { m_walltime_timer.start(); }

void measure_cpu_only_base::record_measurements()
{
  // Update and record timers and counters:
  const auto cur_cpu_time = m_cpu_timer.get_duration();

  m_min_cpu_time = std::min(m_min_cpu_time, cur_cpu_time);
  m_max_cpu_time = std::max(m_max_cpu_time, cur_cpu_time);
  m_total_cpu_time += cur_cpu_time;
  m_cpu_times.push_back(cur_cpu_time);

  ++m_total_samples;

  m_stopping_criterion.add_measurement(cur_cpu_time);
}

bool measure_cpu_only_base::is_finished()
{
  if (m_run_once)
  {
    return true;
  }

  // Check that we've gathered enough samples:
  if (m_total_samples > m_min_samples)
  {
    if (m_stopping_criterion.is_finished())
    {
      return true;
    }
  }

  // Check for timeouts:
  m_walltime_timer.stop();
  if (m_walltime_timer.get_duration() > m_timeout)
  {
    m_max_time_exceeded = true;
    return true;
  }

  return false;
}

void measure_cpu_only_base::run_trials_epilogue() { m_walltime_timer.stop(); }

void measure_cpu_only_base::generate_summaries()
{
  {
    auto &summ = m_state.add_summary("nv/cpu_only/sample_size");
    summ.set_string("name", "Samples");
    summ.set_string("hint", "sample_size");
    summ.set_string("description", "Number of isolated kernel executions");
    summ.set_int64("value", m_total_samples);
  }

  {
    auto &summ = m_state.add_summary("nv/cpu_only/time/cpu/min");
    summ.set_string("name", "Min CPU Time");
    summ.set_string("hint", "duration");
    summ.set_string("description", "Fastest CPU time of isolated kernel executions");
    summ.set_float64("value", m_min_cpu_time);
    summ.set_string("hide", "Hidden by default.");
  }

  {
    auto &summ = m_state.add_summary("nv/cpu_only/time/cpu/max");
    summ.set_string("name", "Max CPU Time");
    summ.set_string("hint", "duration");
    summ.set_string("description", "Slowest CPU time of isolated kernel executions");
    summ.set_float64("value", m_max_cpu_time);
    summ.set_string("hide", "Hidden by default.");
  }

  const auto d_samples = static_cast<nvbench::float64_t>(m_total_samples);
  const auto cpu_mean  = m_total_cpu_time / d_samples;
  {
    auto &summ = m_state.add_summary("nv/cpu_only/time/cpu/mean");
    summ.set_string("name", "CPU Time");
    summ.set_string("hint", "duration");
    summ.set_string("description", "Mean CPU time of isolated kernel executions");
    summ.set_float64("value", cpu_mean);
  }

  const auto cpu_stdev = nvbench::detail::statistics::standard_deviation(m_cpu_times.cbegin(),
                                                                         m_cpu_times.cend(),
                                                                         cpu_mean);
  {
    auto &summ = m_state.add_summary("nv/cpu_only/time/cpu/stdev/absolute");
    summ.set_string("name", "Noise");
    summ.set_string("hint", "percentage");
    summ.set_string("description", "Relative standard deviation of isolated CPU times");
    summ.set_float64("value", cpu_stdev);
    summ.set_string("hide", "Hidden by default.");
  }

  const auto cpu_noise = cpu_stdev / cpu_mean;
  {
    auto &summ = m_state.add_summary("nv/cpu_only/time/cpu/stdev/relative");
    summ.set_string("name", "Noise");
    summ.set_string("hint", "percentage");
    summ.set_string("description", "Relative standard deviation of isolated CPU times");
    summ.set_float64("value", cpu_noise);
  }

  if (const auto items = m_state.get_element_count(); items != 0)
  {
    auto &summ = m_state.add_summary("nv/cpu_only/bw/item_rate");
    summ.set_string("name", "Elem/s");
    summ.set_string("hint", "item_rate");
    summ.set_string("description", "Number of input elements processed per second");
    summ.set_float64("value", static_cast<double>(items) / cpu_mean);
  }

  if (const auto bytes = m_state.get_global_memory_rw_bytes(); bytes != 0)
  {
    const auto avg_used_gmem_bw = static_cast<double>(bytes) / cpu_mean;
    {
      auto &summ = m_state.add_summary("nv/cpu_only/bw/global/bytes_per_second");
      summ.set_string("name", "GlobalMem BW");
      summ.set_string("hint", "byte_rate");
      summ.set_string("description", "Number of bytes read/written per second.");
      summ.set_float64("value", avg_used_gmem_bw);
    }
  } // bandwidth

  {
    auto &summ = m_state.add_summary("nv/cpu_only/walltime");
    summ.set_string("name", "Walltime");
    summ.set_string("hint", "duration");
    summ.set_string("description", "Walltime used for isolated measurements");
    summ.set_float64("value", m_walltime_timer.get_duration());
    summ.set_string("hide", "Hidden by default.");
  }

  // Log if a printer exists:
  if (auto printer_opt_ref = m_state.get_benchmark().get_printer(); printer_opt_ref.has_value())
  {
    auto &printer = printer_opt_ref.value().get();

    if (m_max_time_exceeded)
    {
      const auto timeout = m_walltime_timer.get_duration();

      auto get_param = [this](std::optional<nvbench::float64_t> &param, const std::string &name) {
        if (m_criterion_params.has_value(name))
        {
          param = m_criterion_params.get_float64(name);
        }
      };

      std::optional<nvbench::float64_t> max_noise;
      get_param(max_noise, "max-noise");

      std::optional<nvbench::float64_t> min_time;
      get_param(min_time, "min-time");

      if (max_noise && cpu_noise > *max_noise)
      {
        printer.log(nvbench::log_level::warn,
                    fmt::format("Current measurement timed out ({:0.2f}s) "
                                "while over noise threshold ({:0.2f}% > "
                                "{:0.2f}%)",
                                timeout,
                                cpu_noise * 100,
                                *max_noise * 100));
      }
      if (m_total_samples < m_min_samples)
      {
        printer.log(nvbench::log_level::warn,
                    fmt::format("Current measurement timed out ({:0.2f}s) "
                                "before accumulating min_samples ({} < {})",
                                timeout,
                                m_total_samples,
                                m_min_samples));
      }
      if (min_time && m_total_cpu_time < *min_time)
      {
        printer.log(nvbench::log_level::warn,
                    fmt::format("Current measurement timed out ({:0.2f}s) "
                                "before accumulating min_time ({:0.2f}s < "
                                "{:0.2f}s)",
                                timeout,
                                m_total_cpu_time,
                                *min_time));
      }
    }

    // Log to stdout:
    printer.log(nvbench::log_level::pass,
                fmt::format("CpuOnly: {:0.6f}ms mean CPU, {:0.2f}s total CPU, "
                            "{:0.2f}s total wall, {}x ",
                            cpu_mean * 1e3,
                            m_total_cpu_time,
                            m_walltime_timer.get_duration(),
                            m_total_samples));

    printer.process_bulk_data(m_state, "nv/cpu_only/sample_times", "sample_times", m_cpu_times);
  }
}

void measure_cpu_only_base::check_skip_time(nvbench::float64_t warmup_time)
{
  if (m_skip_time > 0. && warmup_time < m_skip_time)
  {
    auto reason = fmt::format("Warmup time did not meet skip_time limit: "
                              "{:0.3f}us < {:0.3f}us.",
                              warmup_time * 1e6,
                              m_skip_time * 1e6);

    m_state.skip(reason);
    NVBENCH_THROW(std::runtime_error, "{}", std::move(reason));
  }
}

} // namespace nvbench::detail
