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
#include <nvbench/detail/measure_hot.cuh>
#include <nvbench/detail/throw.cuh>
#include <nvbench/device_info.cuh>
#include <nvbench/printer_base.cuh>
#include <nvbench/state.cuh>
#include <nvbench/summary.cuh>

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <utility>
#include <variant>

namespace nvbench::detail
{

measure_hot_base::measure_hot_base(state &exec_state)
    : m_state{exec_state}
    , m_launch{exec_state.get_cuda_stream()}
    , m_min_samples{exec_state.get_min_samples()}
    , m_batch_target_time{exec_state.get_batch_target_time()}
    , m_skip_time{exec_state.get_skip_time()}
    , m_timeout{exec_state.get_timeout()}
{
  try
  {
    nvbench::int64_t cold_samples = m_state.get_summary("nv/cold/sample_size").get_int64("value");
    // Since cold measures converge to a stable result, increase the min_samples
    // to match the cold result if available.
    m_min_samples = std::max(m_min_samples, cold_samples);

    // If the cold measurement ran successfully, disable skip_time. It'd just
    // be annoying to skip now.
    m_skip_time = -1;
  }
  catch (...)
  {
    // If the above threw an exception, we don't have a cold measurement to use.
  }
}

nvbench::int64_t measure_hot_base::predict_batch_size(nvbench::float64_t target_time,
                                                      nvbench::float64_t time_estimate,
                                                      nvbench::int64_t fallback_batch_size)
{
  const auto fallback = std::max(fallback_batch_size, nvbench::int64_t{1});
  if (!std::isfinite(target_time) || target_time <= nvbench::float64_t{0} ||
      !std::isfinite(time_estimate) || time_estimate <= nvbench::float64_t{0})
  {
    return fallback;
  }

  const auto predicted_size = target_time / time_estimate;
  if (!std::isfinite(predicted_size) || predicted_size <= static_cast<nvbench::float64_t>(fallback))
  {
    return fallback;
  }

  if (predicted_size >=
      static_cast<nvbench::float64_t>(std::numeric_limits<nvbench::int64_t>::max()))
  {
    return fallback;
  }

  return static_cast<nvbench::int64_t>(predicted_size);
}

nvbench::int64_t measure_hot_base::grow_batch_size(nvbench::int64_t batch_size,
                                                   nvbench::int64_t minimum_batch_size)
{
  const auto fallback           = std::max(minimum_batch_size, nvbench::int64_t{1});
  const auto batch              = std::max(batch_size, fallback);
  constexpr auto max_batch_size = std::numeric_limits<nvbench::int64_t>::max();
  if (batch > max_batch_size / nvbench::int64_t{2})
  {
    return max_batch_size;
  }

  return std::max(batch * nvbench::int64_t{2}, fallback);
}

void measure_hot_base::check()
{
  const auto device = m_state.get_device();
  if (!device)
  {
    NVBENCH_THROW(std::runtime_error, "{}", "Device required for `hot` measurement.");
  }
  if (!device->is_active())
  { // This means something went wrong higher up. Throw an error.
    NVBENCH_THROW(std::runtime_error, "{}", "Internal error: Current device is not active.");
  }
}

void measure_hot_base::generate_summaries()
{
  const auto d_samples = static_cast<double>(m_total_samples);
  {
    auto &summ = m_state.add_summary("nv/batch/sample_size");
    summ.set_string("name", "Samples");
    summ.set_string("hint", "sample_size");
    summ.set_string("description", "Number of batch kernel executions");
    summ.set_int64("value", m_total_samples);
  }

  const auto avg_cuda_time = m_total_cuda_time / d_samples;
  {
    auto &summ = m_state.add_summary("nv/batch/time/gpu/mean");
    summ.set_string("name", "Batch GPU");
    summ.set_string("hint", "duration");
    summ.set_string("description",
                    "Mean batch kernel execution time "
                    "(measured by CUDA events)");
    summ.set_float64("value", avg_cuda_time);
  }

  {
    auto &summ = m_state.add_summary("nv/batch/walltime");
    summ.set_string("name", "Walltime");
    summ.set_string("hint", "duration");
    summ.set_string("description", "Walltime used for batch measurements");
    summ.set_float64("value", m_walltime_timer.get_duration());
    summ.set_string("hide", "Hidden by default.");
  }

  // Log if a printer exists:
  if (auto printer_ptr = m_state.get_benchmark().get_printer())
  {
    auto &printer = *printer_ptr;

    // Warn if timed out:
    if (m_max_time_exceeded)
    {
      const auto timeout = m_walltime_timer.get_duration();

      if (m_total_samples < m_min_samples)
      {
        printer.log(nvbench::log_level::warn,
                    fmt::format("Current measurement timed out ({:0.2f}s) "
                                "before accumulating min_samples ({} < {})",
                                timeout,
                                m_total_samples,
                                m_min_samples));
      }
      if (m_total_cuda_time < m_batch_target_time)
      {
        printer.log(nvbench::log_level::warn,
                    fmt::format("Current measurement timed out ({:0.2f}s) "
                                "before accumulating batch target time ({:0.2f}s < "
                                "{:0.2f}s)",
                                timeout,
                                m_total_cuda_time,
                                m_batch_target_time));
      }
    }

    // Log to stdout:
    printer.log(nvbench::log_level::pass,
                fmt::format("Batch: {:0.6f}ms GPU, {:0.2f}s total GPU, "
                            "{:0.2f}s total wall, {}x",
                            avg_cuda_time * 1e3,
                            m_total_cuda_time,
                            m_walltime_timer.get_duration(),
                            m_total_samples));
  }
}

void measure_hot_base::check_skip_time(nvbench::float64_t warmup_time)
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

void measure_hot_base::block_stream()
{
  m_blocker.block(m_launch.get_stream(), m_state.get_blocking_kernel_timeout());
}

} // namespace nvbench::detail
