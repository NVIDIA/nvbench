#pragma once

#ifndef NVBENCH_STATE_EXEC_GUARD
#error "This is a private implementation header for state.cuh. " \
       "Do not include it directly."
#endif // NVBENCH_STATE_EXEC_GUARD

#include <nvbench/exec_tag.cuh>
#include <nvbench/state.cuh>

#include <nvbench/detail/measure_cold.cuh>
#include <nvbench/detail/measure_hot.cuh>

#include <type_traits>

namespace nvbench
{

template <typename ExecTags, typename KernelLauncher>
void state::exec(ExecTags tags, KernelLauncher &&kernel_launcher)
{
  using namespace nvbench::exec_tag::impl;
  static_assert(is_exec_tag_v<ExecTags>,
                "`ExecTags` argument must be a member (or combination of "
                "members) from nvbench::exec_tag.");
  if constexpr (!(tags & measure_mask))
  { // No measurements requested -- add the default and rerun:
    this->exec(tags | nvbench::exec_tag::default_tag,
               std::forward<KernelLauncher>(kernel_launcher));
    return;
  }

  if (this->is_skipped())
  {
    return;
  }

  static_assert(!(tags & timer), "Manual timer mode not implemented.");
  static_assert(!(tags & cpu), "CPU-only measurements not implemented.");

  using KL = std::remove_reference_t<KernelLauncher>;
  constexpr auto modifiers = (tags & modifier_mask).flags;

  // Each measurement is deliberately isolated in constexpr branches to
  // avoid instantiating unused measurements.
  if constexpr (tags & cold)
  {
    using measure_t = nvbench::detail::measure_cold<KL, modifiers>;
    measure_t measure{*this, kernel_launcher};
    measure();
  }

  if constexpr (tags & hot)
  {
    using measure_t = nvbench::detail::measure_hot<KL, modifiers>;
    measure_t measure{*this, kernel_launcher};
    measure();
  }
}

} // namespace nvbench
