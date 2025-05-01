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

#pragma once

#ifndef NVBENCH_STATE_EXEC_GUARD
#error "This is a private implementation header for state.cuh. " \
       "Do not include it directly."
#endif // NVBENCH_STATE_EXEC_GUARD

#include <nvbench/config.cuh>
#include <nvbench/detail/kernel_launcher_timer_wrapper.cuh>
#include <nvbench/detail/measure_cold.cuh>
#include <nvbench/detail/measure_cpu_only.cuh>
#include <nvbench/detail/measure_hot.cuh>
#include <nvbench/exec_tag.cuh>
#include <nvbench/state.cuh>

#ifdef NVBENCH_HAS_CUPTI
#include <nvbench/detail/measure_cupti.cuh>
#endif // NVBENCH_HAS_CUPTI

#include <type_traits>

namespace nvbench
{

template <typename ExecTags, typename KernelLauncher>
void state::exec(ExecTags tags, KernelLauncher &&kernel_launcher)
{
  using KL = typename std::remove_reference<KernelLauncher>::type;
  using namespace nvbench::exec_tag::impl;

  static_assert(is_exec_tag_v<ExecTags>,
                "`ExecTags` argument must be a member (or combination of members) from "
                "`nvbench::exec_tag`.");
  static_assert(!((tags & gpu) && (tags & no_gpu)),
                "`nvbench::exec_tag::gpu` and `nvbench::exec_tag::no_gpu` are mutually "
                "exclusive.");

  constexpr auto modifier_tags = tags & modifier_mask;
  constexpr auto measure_tags  = tags & measure_mask;

  // If no measurements selected, pick some defaults based on the modifiers:
  if constexpr (!measure_tags)
  {
    if constexpr (modifier_tags & gpu)
    {
      if constexpr (modifier_tags & no_batch)
      {
        this->exec(cold | modifier_tags, std::forward<KernelLauncher>(kernel_launcher));
      }
      else
      {
        this->exec(cold | hot | modifier_tags, std::forward<KernelLauncher>(kernel_launcher));
      }
    }
    else if constexpr (modifier_tags & no_gpu)
    {
      this->exec(cpu_only | modifier_tags, std::forward<KernelLauncher>(kernel_launcher));
    }
    else // Instantiate both CPU and GPU measurement code:
    {
      if constexpr (modifier_tags & no_batch)
      {
        this->exec(cold | cpu_only | modifier_tags, std::forward<KernelLauncher>(kernel_launcher));
      }
      else
      {
        this->exec(cold | hot | cpu_only | modifier_tags,
                   std::forward<KernelLauncher>(kernel_launcher));
      }
    }
    return;
  }

  if ((modifier_tags & no_gpu) && !this->get_is_cpu_only())
  {
    throw std::runtime_error("The `nvbench::exec_tag::no_gpu` tag requires that "
                             "`set_is_cpu_only(true)` is called when defining the benchmark.");
  }

  if ((modifier_tags & gpu) && this->get_is_cpu_only())
  {
    throw std::runtime_error("The `nvbench::exec_tag::gpu` tag requires that "
                             "`set_is_cpu_only(true)` is NOT called when defining the benchmark.");
  }

  if constexpr (modifier_tags & sync)
  {
    // Syncing will cause the blocking kernel pattern to deadlock:
    this->set_disable_blocking_kernel(true);
    // Syncing will cause the throttle frequency measurements to be skewed heavily:
    this->set_throttle_threshold(0.f);
  }

  if (this->is_skipped())
  {
    return;
  }

  if (this->get_is_cpu_only())
  {
    if constexpr (tags & cpu_only) // Prevent instantiation when not needed
    {
      static_assert(!(tags & gpu), "CPU-only measurement doesn't support the `gpu` exec_tag.");

      if constexpr (tags & timer)
      {
        using measure_t = nvbench::detail::measure_cpu_only<KL>;
        measure_t measure{*this, kernel_launcher};
        measure();
      }
      else
      { // Need to wrap the kernel launcher with a timer wrapper:
        using wrapper_t = nvbench::detail::kernel_launch_timer_wrapper<KL>;
        wrapper_t wrapper{kernel_launcher};

        using measure_t = nvbench::detail::measure_cpu_only<wrapper_t>;
        measure_t measure(*this, wrapper);
        measure();
      }
    }
  }
  else
  {
    if constexpr (tags & cold) // Prevent instantiation when not needed
    {
      static_assert(!(tags & no_gpu), "Cold measurement doesn't support the `no_gpu` exec_tag.");

      if constexpr (tags & timer)
      {
#ifdef NVBENCH_HAS_CUPTI
        if (this->is_cupti_required() && !this->get_run_once())
        {
          using measure_t = nvbench::detail::measure_cupti<KL>;
          measure_t measure{*this, kernel_launcher};
          measure();
        }
#endif

        using measure_t = nvbench::detail::measure_cold<KL>;
        measure_t measure{*this, kernel_launcher};
        measure();
      }
      else
      { // Need to wrap the kernel launcher with a timer wrapper:
        using wrapper_t = nvbench::detail::kernel_launch_timer_wrapper<KL>;
        wrapper_t wrapper{kernel_launcher};

#ifdef NVBENCH_HAS_CUPTI
        if (this->is_cupti_required() && !this->get_run_once())
        {
          using measure_t = nvbench::detail::measure_cupti<wrapper_t>;
          measure_t measure{*this, wrapper};
          measure();
        }
#endif

        using measure_t = nvbench::detail::measure_cold<wrapper_t>;
        measure_t measure(*this, wrapper);
        measure();
      }
    }

    if constexpr (tags & hot) // Prevent instantiation when not needed
    {
      static_assert(!(tags & sync), "Hot measurement doesn't support the `sync` exec_tag.");
      static_assert(!(tags & timer), "Hot measurement doesn't support the `timer` exec_tag.");
      static_assert(!(tags & no_batch), "Hot measurement doesn't support the `no_batch` exec_tag.");
      static_assert(!(tags & no_gpu), "Hot measurement doesn't support the `no_gpu` exec_tag.");

      if (!this->get_run_once())
      {
        using measure_t = nvbench::detail::measure_hot<KL>;
        measure_t measure{*this, kernel_launcher};
        measure();
      }
    }
  }
}

} // namespace nvbench
