#pragma once

namespace nvbench
{

struct cuda_stream;


/**
 * Blocks a CUDA stream -- many sharp edges, read docs carefully.
 *
 * @warning This helper breaks the CUDA programming model and will cause
 * deadlocks if misused. It should not be used outside of benchmarking.
 * See caveats section below.
 *
 * This is used to improve the precision of timing with CUDA events. Consider
 * the following pattern for timing a kernel launch:
 *
 * ```
 * NVBENCH_CUDA_CALL(cudaEventRecord(start_event));
 * my_kernel<<<...>>>();
 * NVBENCH_CUDA_CALL(cudaEventRecord(stop_event));
 * ```
 *
 * The `start_event` may be recorded a non-trivial amount of time before
 * `my_kernel` is ready to launch due to various work submission latencies. To
 * reduce the impact of these latencies, blocking_kernel can be used to prevent
 * the `start_event` from being recorded until all work is queued:
 *
 * ```
 * blocking_kernel blocker;
 * blocker.block(stream);
 *
 * NVBENCH_CUDA_CALL(cudaEventRecord(start_event));
 * my_kernel<<<...>>>();
 * NVBENCH_CUDA_CALL(cudaEventRecord(stop_event))
 *
 * blocker.release();
 * ```
 *
 * The work submitted after `blocker.block(stream)` will not execute until
 * `blocker.release()` is called.
 *
 * ## Caveats and warnings
 *
 * - Every call to `block()` must be followed by a call to `release()`.
 * - Do not queue "too much" work while blocking.
 *   - Amount of work depends on device and driver.
 *   - Do tests and schedule conservatively (~32 kernel launches max).
 * - This helper does NOT guarantee that the work submitted while blocking will
 *   execute uninterrupted.
 *   - Kernels on other streams may run between the `cudaEventRecord` calls
 *     in the above example.
 */
struct blocking_kernel
{
  blocking_kernel();
  ~blocking_kernel();

  void block(const nvbench::cuda_stream &stream);
  void release();

  // move-only
  blocking_kernel(const blocking_kernel &) = delete;
  blocking_kernel(blocking_kernel &&)      = default;
  blocking_kernel &operator=(const blocking_kernel &) = delete;
  blocking_kernel &operator=(blocking_kernel &&) = default;

private:
  int m_host_flag{};
  int *m_device_flag{};
};

} // namespace nvbench
