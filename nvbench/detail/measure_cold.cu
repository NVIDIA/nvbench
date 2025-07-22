/*
 *  Copyright 2021-2022 NVIDIA Corporation
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
#include <nvbench/detail/measure_cold.cuh>
#include <nvbench/detail/throw.cuh>
#include <nvbench/device_info.cuh>
#include <nvbench/printer_base.cuh>
#include <nvbench/state.cuh>
#include <nvbench/summary.cuh>

#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <limits>
#include <optional>
#include <thread>

namespace nvbench::detail
{

measure_cold_base::measure_cold_base(state &exec_state)
    : m_state{exec_state}
    , m_launch{exec_state.get_cuda_stream()}
    , m_criterion_params{exec_state.get_criterion_params()}
    , m_stopping_criterion{nvbench::criterion_manager::get().get_criterion(
        exec_state.get_stopping_criterion())}
    , m_disable_blocking_kernel{exec_state.get_disable_blocking_kernel()}
    , m_run_once{exec_state.get_run_once()}
    , m_check_throttling(!exec_state.get_run_once() && exec_state.get_throttle_threshold() > 0.f)
    , m_min_samples{exec_state.get_min_samples()}
    , m_skip_time{exec_state.get_skip_time()}
    , m_timeout{exec_state.get_timeout()}
    , m_throttle_threshold(exec_state.get_throttle_threshold())
    , m_throttle_recovery_delay(exec_state.get_throttle_recovery_delay())
{
  if (m_min_samples > 0)
  {
    m_cuda_times.reserve(static_cast<std::size_t>(m_min_samples));
    m_cpu_times.reserve(static_cast<std::size_t>(m_min_samples));
  }
}

void measure_cold_base::check()
{
  const auto device = m_state.get_device();
  if (!device)
  {
    NVBENCH_THROW(std::runtime_error, "{}", "Device required for `cold` measurement.");
  }
  if (!device->is_active())
  { // This means something went wrong higher up. Throw an error.
    NVBENCH_THROW(std::runtime_error, "{}", "Internal error: Current device is not active.");
  }
}

void measure_cold_base::initialize()
{
  m_min_cuda_time             = std::numeric_limits<nvbench::float64_t>::max();
  m_max_cuda_time             = std::numeric_limits<nvbench::float64_t>::lowest();
  m_total_cuda_time           = 0.;
  m_min_cpu_time              = std::numeric_limits<nvbench::float64_t>::max();
  m_max_cpu_time              = std::numeric_limits<nvbench::float64_t>::lowest();
  m_total_cpu_time            = 0.;
  m_sm_clock_rate_accumulator = 0.;
  m_total_samples             = 0;
  m_max_time_exceeded         = false;

  m_dynamic_throttle_recovery_delay = m_throttle_recovery_delay;
  m_throttle_discard_count          = 0;

  m_cuda_times.clear();
  m_cpu_times.clear();

  m_stopping_criterion.initialize(m_criterion_params);
}

void measure_cold_base::run_trials_prologue() { m_walltime_timer.start(); }

void measure_cold_base::record_measurements()
{
  if (m_check_throttling)
  {
    const auto current_clock_rate = m_gpu_frequency.get_clock_frequency();
    const auto default_clock_rate =
      static_cast<float>(m_state.get_device()->get_sm_default_clock_rate());

    if (m_gpu_frequency.has_throttled(default_clock_rate, m_throttle_threshold))
    {
      if (m_throttle_discard_count > 2)
      {
        // Throttling detected in multiple consecutive trials. The delay is not sufficient to
        // recover. Increase the delay by no more than half of a second:
        m_dynamic_throttle_recovery_delay += std::min(m_dynamic_throttle_recovery_delay * 1.5f,
                                                      0.5f);
      }

      if (auto printer_opt_ref = m_state.get_benchmark().get_printer(); printer_opt_ref.has_value())
      {
        auto &printer = printer_opt_ref.value().get();
        printer.log(nvbench::log_level::warn,
                    fmt::format("GPU throttled below threshold ({:0.2f} MHz / {:0.2f} MHz) "
                                "({:0.0f}% < {:0.0f}%) on sample {}. Discarding previous trial "
                                "and pausing for {:0.3f}s.",
                                current_clock_rate / 1000000.0f,
                                default_clock_rate / 1000000.0f,
                                100.0f * (current_clock_rate / default_clock_rate),
                                100.0f * m_throttle_threshold,
                                m_total_samples,
                                m_dynamic_throttle_recovery_delay));
      }

      if (m_dynamic_throttle_recovery_delay > 0.0f)
      { // let the GPU cool down
        std::this_thread::sleep_for(
          std::chrono::duration<float>(m_dynamic_throttle_recovery_delay));
      }

      m_throttle_discard_count += 1;

      // ignore this measurement
      return;
    }
    m_throttle_discard_count = 0;

    m_sm_clock_rate_accumulator += current_clock_rate;
  }

  // Update and record timers and counters:
  const auto cur_cuda_time = m_cuda_timer.get_duration();
  const auto cur_cpu_time  = m_cpu_timer.get_duration();

  m_min_cuda_time = std::min(m_min_cuda_time, cur_cuda_time);
  m_max_cuda_time = std::max(m_max_cuda_time, cur_cuda_time);
  m_total_cuda_time += cur_cuda_time;
  m_cuda_times.push_back(cur_cuda_time);

  m_min_cpu_time = std::min(m_min_cpu_time, cur_cpu_time);
  m_max_cpu_time = std::max(m_max_cpu_time, cur_cpu_time);
  m_total_cpu_time += cur_cpu_time;
  m_cpu_times.push_back(cur_cpu_time);

  ++m_total_samples;

  m_stopping_criterion.add_measurement(cur_cuda_time);
}

bool measure_cold_base::is_finished()
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

void measure_cold_base::run_trials_epilogue() { m_walltime_timer.stop(); }

void measure_cold_base::generate_summaries()
{
  {
    auto &summ = m_state.add_summary("nv/cold/sample_size");
    summ.set_string("name", "Samples");
    summ.set_string("hint", "sample_size");
    summ.set_string("description", "Number of isolated kernel executions");
    summ.set_int64("value", m_total_samples);
  }

  {
    auto &summ = m_state.add_summary("nv/cold/time/cpu/min");
    summ.set_string("name", "Min CPU Time");
    summ.set_string("hint", "duration");
    summ.set_string("description",
                    "Fastest isolated kernel execution time "
                    "(measured on host CPU)");
    summ.set_float64("value", m_min_cpu_time);
    summ.set_string("hide", "Hidden by default.");
  }

  {
    auto &summ = m_state.add_summary("nv/cold/time/cpu/max");
    summ.set_string("name", "Max CPU Time");
    summ.set_string("hint", "duration");
    summ.set_string("description",
                    "Slowest isolated kernel execution time "
                    "(measured on host CPU)");
    summ.set_float64("value", m_max_cpu_time);
    summ.set_string("hide", "Hidden by default.");
  }

  const auto d_samples = static_cast<double>(m_total_samples);
  const auto cpu_mean  = m_total_cpu_time / d_samples;
  {
    auto &summ = m_state.add_summary("nv/cold/time/cpu/mean");
    summ.set_string("name", "CPU Time");
    summ.set_string("hint", "duration");
    summ.set_string("description",
                    "Mean isolated kernel execution time "
                    "(measured on host CPU)");
    summ.set_float64("value", cpu_mean);
  }

  const auto cpu_stdev = nvbench::detail::statistics::standard_deviation(m_cpu_times.cbegin(),
                                                                         m_cpu_times.cend(),
                                                                         cpu_mean);
  {
    auto &summ = m_state.add_summary("nv/cold/time/cpu/stdev/absolute");
    summ.set_string("name", "Noise");
    summ.set_string("hint", "percentage");
    summ.set_string("description", "Standard deviation of isolated CPU times");
    summ.set_float64("value", cpu_stdev);
    summ.set_string("hide", "Hidden by default.");
  }

  const auto cpu_noise = cpu_stdev / cpu_mean;
  {
    auto &summ = m_state.add_summary("nv/cold/time/cpu/stdev/relative");
    summ.set_string("name", "Noise");
    summ.set_string("hint", "percentage");
    summ.set_string("description", "Relative standard deviation of isolated CPU times");
    summ.set_float64("value", cpu_noise);
  }

  {
    auto &summ = m_state.add_summary("nv/cold/time/gpu/min");
    summ.set_string("name", "Min GPU Time");
    summ.set_string("hint", "duration");
    summ.set_string("description",
                    "Fastest isolated kernel execution time "
                    "(measured with CUDA events)");
    summ.set_float64("value", m_min_cuda_time);
    summ.set_string("hide", "Hidden by default.");
  }

  {
    auto &summ = m_state.add_summary("nv/cold/time/gpu/max");
    summ.set_string("name", "Max GPU Time");
    summ.set_string("hint", "duration");
    summ.set_string("description",
                    "Slowest isolated kernel execution time "
                    "(measured with CUDA events)");
    summ.set_float64("value", m_max_cuda_time);
    summ.set_string("hide", "Hidden by default.");
  }

  const auto cuda_mean = m_total_cuda_time / d_samples;
  {
    auto &summ = m_state.add_summary("nv/cold/time/gpu/mean");
    summ.set_string("name", "GPU Time");
    summ.set_string("hint", "duration");
    summ.set_string("description",
                    "Mean isolated kernel execution time "
                    "(measured with CUDA events)");
    summ.set_float64("value", cuda_mean);
  }

  const auto cuda_stdev = nvbench::detail::statistics::standard_deviation(m_cuda_times.cbegin(),
                                                                          m_cuda_times.cend(),
                                                                          cuda_mean);
  {
    auto &summ = m_state.add_summary("nv/cold/time/gpu/stdev/absolute");
    summ.set_string("name", "Noise");
    summ.set_string("hint", "percentage");
    summ.set_string("description", "Relative standard deviation of isolated GPU times");
    summ.set_float64("value", cuda_stdev);
    summ.set_string("hide", "Hidden by default.");
  }

  const auto cuda_noise = cuda_stdev / cuda_mean;
  {
    auto &summ = m_state.add_summary("nv/cold/time/gpu/stdev/relative");
    summ.set_string("name", "Noise");
    summ.set_string("hint", "percentage");
    summ.set_string("description", "Relative standard deviation of isolated GPU times");
    summ.set_float64("value", cuda_noise);
  }

  if (const auto items = m_state.get_element_count(); items != 0)
  {
    auto &summ = m_state.add_summary("nv/cold/bw/item_rate");
    summ.set_string("name", "Elem/s");
    summ.set_string("hint", "item_rate");
    summ.set_string("description", "Number of input elements processed per second");
    summ.set_float64("value", static_cast<double>(items) / cuda_mean);
  }

  if (const auto bytes = m_state.get_global_memory_rw_bytes(); bytes != 0)
  {
    const auto avg_used_gmem_bw = static_cast<double>(bytes) / cuda_mean;
    {
      auto &summ = m_state.add_summary("nv/cold/bw/global/bytes_per_second");
      summ.set_string("name", "GlobalMem BW");
      summ.set_string("hint", "byte_rate");
      summ.set_string("description",
                      "Number of bytes read/written per second to the CUDA "
                      "device's global memory");
      summ.set_float64("value", avg_used_gmem_bw);
    }

    {
      const auto peak_gmem_bw =
        static_cast<double>(m_state.get_device()->get_global_memory_bus_bandwidth());

      auto &summ = m_state.add_summary("nv/cold/bw/global/utilization");
      summ.set_string("name", "BWUtil");
      summ.set_string("hint", "percentage");
      summ.set_string("description",
                      "Global device memory utilization as a percentage of the "
                      "device's peak bandwidth");
      summ.set_float64("value", avg_used_gmem_bw / peak_gmem_bw);
    }
  } // bandwidth

  {
    auto &summ = m_state.add_summary("nv/cold/walltime");
    summ.set_string("name", "Walltime");
    summ.set_string("hint", "duration");
    summ.set_string("description", "Walltime used for isolated measurements");
    summ.set_float64("value", m_walltime_timer.get_duration());
    summ.set_string("hide", "Hidden by default.");
  }

  if (m_sm_clock_rate_accumulator != 0.)
  {
    const auto clock_mean = m_sm_clock_rate_accumulator / d_samples;

    {
      auto &summ = m_state.add_summary("nv/cold/sm_clock_rate/mean");
      summ.set_string("name", "Clock Rate");
      summ.set_string("hint", "frequency");
      summ.set_string("description", "Mean SM clock rate");
      summ.set_string("hide", "Hidden by default.");
      summ.set_float64("value", clock_mean);
    }

    {
      const auto default_clock_rate =
        static_cast<nvbench::float64_t>(m_state.get_device()->get_sm_default_clock_rate());

      auto &summ = m_state.add_summary("nv/cold/sm_clock_rate/scaling/percent");
      summ.set_string("name", "Clock Scaling");
      summ.set_string("hint", "percentage");
      summ.set_string("description", "Mean SM clock rate as a percentage of default clock rate.");
      summ.set_string("hide", "Hidden by default.");
      summ.set_float64("value", clock_mean / default_clock_rate);
    }
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

      if (max_noise && cuda_noise > *max_noise)
      {
        printer.log(nvbench::log_level::warn,
                    fmt::format("Current measurement timed out ({:0.2f}s) "
                                "while over noise threshold ({:0.2f}% > "
                                "{:0.2f}%)",
                                timeout,
                                cuda_noise * 100,
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
      if (min_time && m_total_cuda_time < *min_time)
      {
        printer.log(nvbench::log_level::warn,
                    fmt::format("Current measurement timed out ({:0.2f}s) "
                                "before accumulating min_time ({:0.2f}s < "
                                "{:0.2f}s)",
                                timeout,
                                m_total_cuda_time,
                                *min_time));
      }
    }

    // Log to stdout:
    printer.log(nvbench::log_level::pass,
                fmt::format("Cold: {:0.6f}ms GPU, {:0.6f}ms CPU, {:0.2f}s "
                            "total GPU, {:0.2f}s total wall, {}x ",
                            cuda_mean * 1e3,
                            cpu_mean * 1e3,
                            m_total_cuda_time,
                            m_walltime_timer.get_duration(),
                            m_total_samples));

    printer.process_bulk_data(m_state, "nv/cold/sample_times", "sample_times", m_cuda_times);
  }
}

void measure_cold_base::check_skip_time(nvbench::float64_t warmup_time)
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

void measure_cold_base::block_stream()
{
  m_blocker.block(m_launch.get_stream(), m_state.get_blocking_kernel_timeout());
}

} // namespace nvbench::detail
