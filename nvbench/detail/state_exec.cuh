/*
 *  Copyright 2020 NVIDIA Corporation
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

#include <nvbench/exec_tag.cuh>
#include <nvbench/state.cuh>

#include <nvbench/detail/kernel_launcher_timer_wrapper.cuh>
#include <nvbench/detail/measure_cold.cuh>
#include <nvbench/detail/measure_hot.cuh>

#include <type_traits>

namespace nvbench
{

template <typename ExecTags, typename KernelLauncher>
void state::exec(ExecTags tags, KernelLauncher &&kernel_launcher)
{
  using KL = typename std::remove_reference<KernelLauncher>::type;
  using namespace nvbench::exec_tag::impl;
  static_assert(is_exec_tag_v<ExecTags>,
                "`ExecTags` argument must be a member (or combination of "
                "members) from nvbench::exec_tag.");

  // If no measurements selected, pick some defaults based on the modifiers:
  constexpr auto measure_tags = tags & measure_mask;
  if constexpr (!measure_tags)
  {
    constexpr auto modifier_tags = tags & modifier_mask;
    if constexpr (modifier_tags & (timer | sync))
    { // Can't do hot timings with manual timer or sync; whole point is to not
      // sync in between executions.
      this->exec(cold | tags, std::forward<KernelLauncher>(kernel_launcher));
    }
    else
    {
      this->exec(cold | hot | tags,
                 std::forward<KernelLauncher>(kernel_launcher));
    }
    return;
  }

  if (this->is_skipped())
  {
    return;
  }

  // Each measurement is deliberately isolated in constexpr branches to
  // avoid instantiating unused measurements.
  if constexpr (tags & cold)
  {
    constexpr bool use_blocking_kernel = !(tags & no_block);
    if constexpr (tags & timer)
    {
      using measure_t = nvbench::detail::measure_cold<KL, use_blocking_kernel>;
      measure_t measure{*this, kernel_launcher};
      measure();
    }
    else
    { // Need to wrap the kernel launcher with a timer wrapper:
      using wrapper_t = nvbench::detail::kernel_launch_timer_wrapper<KL>;
      using measure_t =
        nvbench::detail::measure_cold<wrapper_t, use_blocking_kernel>;
      wrapper_t wrapper{kernel_launcher};
      measure_t measure(*this, wrapper);
      measure();
    }
  }

  if constexpr (tags & hot)
  {
    static_assert(!(tags & sync),
                  "Hot measurement doesn't support the `sync` exec_tag.");
    static_assert(!(tags & timer),
                  "Hot measurement doesn't support the `timer` exec_tag.");
    constexpr bool use_blocking_kernel = !(tags & no_block);
    using measure_t = nvbench::detail::measure_hot<KL, use_blocking_kernel>;
    measure_t measure{*this, kernel_launcher};
    measure();
  }
}

} // namespace nvbench
