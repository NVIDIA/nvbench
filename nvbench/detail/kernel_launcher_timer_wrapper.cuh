#pragma once

namespace nvbench
{

struct launch;

namespace detail
{

/// Wrap a KernelLauncher call between start/stop calls.
///
/// This simplifies the implementation of measurements that support
/// `nvbench::exec_tag::timer`.
template <typename KernelLauncher>
struct kernel_launch_timer_wrapper
{
 explicit kernel_launch_timer_wrapper(KernelLauncher &launcher)
      : m_kernel_launcher{launcher}
  {}

  template <typename TimerT>
  __forceinline__ void operator()(nvbench::launch &launch, TimerT &timer)
  {
    timer.start();
    m_kernel_launcher(launch);
    timer.stop();
  }

  KernelLauncher &m_kernel_launcher;
};

} // namespace detail
} // namespace nvbench
