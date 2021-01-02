#pragma once

#include <nvbench/detail/measure_cold.cuh>
#include <nvbench/detail/measure_hot.cuh>

namespace nvbench
{

template <typename KernelLauncher>
void exec(nvbench::state &exec_state, KernelLauncher &&kernel_launcher)
{
  using KL = std::remove_reference_t<KernelLauncher>;

  {
    nvbench::detail::measure_cold<KL> cold{exec_state, kernel_launcher};
    cold();
  }

  {
    nvbench::detail::measure_hot<KL> hot{exec_state, kernel_launcher};
    hot();
  }
}

} // namespace nvbench
