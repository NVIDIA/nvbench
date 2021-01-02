#include <nvbench/detail/measure_hot.cuh>

#include <nvbench/benchmark_base.cuh>
#include <nvbench/state.cuh>
#include <nvbench/summary.cuh>

#include <fmt/format.h>

#include <cstdio>
#include <variant>

// note that these can be removed once there's a device_manager or some such:
#include <cuda_runtime_api.h>
#include <nvbench/cuda_call.cuh>

namespace nvbench
{

namespace detail
{

void measure_hot_base::generate_summaries()
{
  {
    auto &summ = m_state.add_summary("Number of Trials (Hot)");
    summ.set_string("short_name", "Hot Trials");
    summ.set_string("description",
                    "Number of kernel executions in hot time measurements.");
    summ.set_int64("value", m_num_trials);
  }

  const auto avg_cuda_time = m_cuda_time / m_num_trials;
  {
    auto &summ = m_state.add_summary("Average GPU Time (Hot)");
    summ.set_string("hint", "duration");
    summ.set_string("short_name", "Hot GPU");
    summ.set_string("description",
                    "Average back-to-back kernel execution time as measured "
                    "by CUDA events.");
    summ.set_float64("value", avg_cuda_time);
  }

  const auto avg_cpu_time = m_cpu_time / m_num_trials;
  {
    auto &summ = m_state.add_summary("Average CPU Time (Hot)");
    summ.set_string("hide",
                    "Usually not interesting; too similar to hot GPU times.");
    summ.set_string("hint", "duration");
    summ.set_string("short_name", "Hot CPU");
    summ.set_string("description",
                    "Average back-to-back kernel execution time observed "
                    "from host.");
    summ.set_float64("value", avg_cpu_time);
  }

  if (const auto items = m_state.get_items_processed_per_launch(); items != 0)
  {
    auto &summ = m_state.add_summary("Item Throughput");
    summ.set_string("hint", "item_rate");
    summ.set_string("short_name", "Item Rate");
    summ.set_string("description", "Number of input items handled per second.");
    summ.set_float64("value", items / avg_cuda_time);
  }

  if (const auto bytes = m_state.get_global_bytes_accessed_per_launch();
      bytes != 0)
  {
    const auto avg_used_gmem_bw = bytes / avg_cuda_time;
    {
      auto &summ = m_state.add_summary("Average Global Memory Throughput");
      summ.set_string("hint", "byte_rate");
      summ.set_string("short_name", "GlobalMemUse");
      summ.set_string("description",
                      "Number of bytes read/written per second to the CUDA "
                      "device's global memory.");
      summ.set_float64("value", avg_used_gmem_bw);
    }

    // TODO cache this in a singleton somewhere.
    int dev_id{};
    cudaDeviceProp prop{};
    NVBENCH_CUDA_CALL(cudaGetDevice(&dev_id));
    NVBENCH_CUDA_CALL(cudaGetDeviceProperties(&prop, dev_id));
    // clock rate in khz, width in bits. Result in bytes/sec.
    const auto peak_gmem_bw = 2 * 1000. * prop.memoryClockRate * // (sec^-1)
                              prop.memoryBusWidth / CHAR_BIT;    // bytes

    {
      auto &summ = m_state.add_summary("Percent Peak Global Memory Throughput");
      summ.set_string("hint", "percentage");
      summ.set_string("short_name", "PeakGMem");
      summ.set_string("description",
                      "Global device memory throughput as a percentage of the "
                      "device's peak bandwidth.");
      summ.set_float64("value", avg_used_gmem_bw / peak_gmem_bw * 100.);
    }
  }

  // Log to stdout:
  fmt::memory_buffer param_buffer;
  fmt::format_to(param_buffer, "");
  const axes_metadata &axes = m_state.get_benchmark().get_axes();
  const auto &axis_values = m_state.get_axis_values();
  for (const auto &name : axis_values.get_names())
  {
    fmt::format_to(param_buffer, "{}=", name);

    // Handle power-of-two int64 axes differently:
    if (axis_values.get_type(name) == named_values::type::int64 &&
        axes.get_int64_axis(name).is_power_of_two())
    {
      const nvbench::uint64_t value    = axis_values.get_int64(name);
      const nvbench::uint64_t exponent = int64_axis::compute_log2(value);
      fmt::format_to(param_buffer, "2^{}", exponent);
    }
    else
    {
      std::visit(
        [&param_buffer](const auto &val) {
          fmt::format_to(param_buffer, "{} ", val);
        },
        axis_values.get_value(name));
    }
  }

  fmt::print("Benchmark {} Params: [ {}] Hot  {:.6f} ms GPU, {:.6f} ms CPU, "
             "{}x\n",
             m_state.get_benchmark().get_name(),
             fmt::to_string(param_buffer),
             avg_cuda_time * 1e3,
             avg_cpu_time * 1e3,
             m_num_trials);
  std::fflush(stdout);
}

} // namespace detail

} // namespace nvbench
