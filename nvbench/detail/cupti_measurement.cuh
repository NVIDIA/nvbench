// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <nvbench/cpu_timer.cuh>
#include <nvbench/cupti_timer.cuh>

namespace nvbench::detail
{

// Warmup measurements need a valid duration for skip a benchmark (e.g. too short) or to estimate
// the number of repetitions to reach target batch duration predictions.
template <typename Measurement>
nvbench::float64_t retry_cupti_warmup(Measurement &&measure, nvbench::float64_t timeout)
{
  constexpr int max_attempts = 8;

  nvbench::cpu_timer walltime;
  walltime.start();
  for (int attempt = 1;; ++attempt)
  {
    try
    {
      return measure();
    }
    catch (const nvbench::cupti_collection_failure &)
    {
      walltime.stop();
      if (attempt == max_attempts || walltime.get_duration() >= timeout)
      {
        throw;
      }
    }
  }
}

} // namespace nvbench::detail
