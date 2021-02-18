#pragma once

#include <nvbench/device_info.cuh>
#include <nvbench/exec_tag.cuh>
#include <nvbench/named_values.cuh>
#include <nvbench/summary.cuh>
#include <nvbench/types.cuh>

#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace nvbench
{

struct benchmark_base;

namespace detail
{
struct state_generator;
struct state_tester;
} // namespace detail

/**
 * Stores all information about a particular benchmark configuration.
 *
 * One state object exists for every combination of a benchmark's parameter
 * axes. It provides access to:
 * - Parameter values (get_int64, get_float64, get_string)
 *   - The names of parameters from type axes are stored as strings.
 * - Skip information (skip, is_skipped, get_skip_reason)
 *   - If the benchmark fails or is invalid, it may be skipped with an
 *     informative message.
 * - Summaries (add_summary, get_summary, get_summaries)
 *   - Summaries store measurement information as key/value pairs.
 *     See nvbench::summary for details.
 */
struct state
{
  // move-only
  state(const state &) = delete;
  state(state &&)      = default;
  state &operator=(const state &) = delete;
  state &operator=(state &&) = default;

  /// The CUDA device associated with with this benchmark state. May be
  /// nullopt for CPU-only benchmarks.
  [[nodiscard]] const std::optional<nvbench::device_info> &get_device() const
  {
    return m_device;
  }

  /// An index into a benchmark::type_configs type_list. Returns 0 if no type
  /// axes in the associated benchmark.
  [[nodiscard]] std::size_t get_type_config_index() const
  {
    return m_type_config_index;
  }

  [[nodiscard]] nvbench::int64_t get_int64(const std::string &axis_name) const;

  [[nodiscard]] nvbench::float64_t
  get_float64(const std::string &axis_name) const;

  [[nodiscard]] const std::string &
  get_string(const std::string &axis_name) const;

  void add_element_count(std::size_t elements, std::string column_name = {});

  void set_element_count(nvbench::int64_t elements)
  {
    m_element_count = elements;
  }
  [[nodiscard]] nvbench::int64_t get_element_count() const
  {
    return m_element_count;
  }

  template <typename ElementType>
  void add_global_memory_reads(std::size_t count, std::string column_name = {})
  {
    this->add_global_memory_reads(count * sizeof(ElementType),
                                  std::move(column_name));
  }
  void add_global_memory_reads(std::size_t bytes, std::string column_name);

  template <typename ElementType>
  void add_global_memory_writes(std::size_t count, std::string column_name = {})
  {
    this->add_global_memory_writes(count * sizeof(ElementType),
                                   std::move(column_name));
  }
  void add_global_memory_writes(std::size_t bytes, std::string column_name);

  void set_global_memory_rw_bytes(nvbench::int64_t bytes)
  {
    m_global_memory_rw_bytes = bytes;
  }
  [[nodiscard]] nvbench::int64_t get_global_memory_rw_bytes() const
  {
    return m_global_memory_rw_bytes;
  }

  void skip(std::string reason) { m_skip_reason = std::move(reason); }
  [[nodiscard]] bool is_skipped() const { return !m_skip_reason.empty(); }
  [[nodiscard]] const std::string &get_skip_reason() const
  {
    return m_skip_reason;
  }

  /// Execute at least this many trials per measurement. @{
  [[nodiscard]] nvbench::int64_t get_min_samples() const
  {
    return m_min_samples;
  }
  void set_min_samples(nvbench::int64_t min_samples)
  {
    m_min_samples = min_samples;
  }
  /// @}

  /// Accumulate at least this many seconds of timing data per measurement. @{
  [[nodiscard]] nvbench::float64_t get_min_time() const { return m_min_time; }
  void set_min_time(nvbench::float64_t min_time) { m_min_time = min_time; }
  /// @}

  /// Specify the maximum amount of noise if a measurement supports noise.
  /// Noise is the relative standard deviation expressed as a percentage:
  /// `noise = 100 * (stdev / mean_time)`. @{
  [[nodiscard]] nvbench::float64_t get_max_noise() const { return m_max_noise; }
  void set_max_noise(nvbench::float64_t max_noise) { m_max_noise = max_noise; }
  /// @}

  /// If a warmup run finishes in less than `skip_time`, the measurement will
  /// be skipped.
  /// Extremely fast kernels (< 5000 ns) often timeout before they can
  /// accumulate `min_time` measurements, and are often uninteresting. Setting
  /// this value can help improve performance by skipping time consuming
  /// measurement that don't provide much information.
  /// Default value is -1., which disables the feature.
  /// @{
  [[nodiscard]] nvbench::float64_t get_skip_time() const { return m_skip_time; }
  void set_skip_time(nvbench::float64_t skip_time) { m_skip_time = skip_time; }
  /// @}

  /// If a measurement take more than `timeout` seconds to complete, stop the
  /// measurement early. A warning should be printed if this happens.
  /// This setting overrides all other termination criteria.
  /// Note that this is measured in CPU walltime, not sample time.
  /// @{
  [[nodiscard]] nvbench::float64_t get_timeout() const { return m_timeout; }
  void set_timeout(nvbench::float64_t timeout) { m_timeout = timeout; }
  /// @}

  [[nodiscard]] const named_values &get_axis_values() const
  {
    return m_axis_values;
  }

  [[nodiscard]] const benchmark_base &get_benchmark() const
  {
    return m_benchmark;
  }

  summary &add_summary(std::string summary_name);
  summary &add_summary(summary s);
  [[nodiscard]] const summary &get_summary(std::string_view name) const;
  [[nodiscard]] summary &get_summary(std::string_view name);
  [[nodiscard]] const std::vector<summary> &get_summaries() const;
  [[nodiscard]] std::vector<summary> &get_summaries();

  /// A single line description of the state:
  ///
  /// ```
  /// <bench_name> [<parameters>]
  /// ```
  [[nodiscard]] std::string get_short_description() const;

  // TODO This will need detailed docs and include a reference to an appropriate
  // section of the user's guide
  template <typename ExecTags, typename KernelLauncher>
  void exec(ExecTags, KernelLauncher &&kernel_launcher);

  template <typename KernelLauncher>
  void exec(KernelLauncher &&kernel_launcher)
  {
    this->exec(nvbench::exec_tag::default_measurements,
               std::forward<KernelLauncher>(kernel_launcher));
  }

private:
  friend struct nvbench::detail::state_generator;
  friend struct nvbench::detail::state_tester;

  explicit state(const benchmark_base &bench);

  state(const benchmark_base &bench,
        nvbench::named_values values,
        std::optional<nvbench::device_info> device,
        std::size_t type_config_index);

  std::reference_wrapper<const nvbench::benchmark_base> m_benchmark;
  nvbench::named_values m_axis_values;
  std::optional<nvbench::device_info> m_device;
  std::size_t m_type_config_index{};

  nvbench::int64_t m_min_samples;
  nvbench::float64_t m_min_time;
  nvbench::float64_t m_max_noise;

  nvbench::float64_t m_skip_time;
  nvbench::float64_t m_timeout;

  std::vector<nvbench::summary> m_summaries;
  std::string m_skip_reason;
  nvbench::int64_t m_element_count{};
  nvbench::int64_t m_global_memory_rw_bytes{};
};

} // namespace nvbench

#define NVBENCH_STATE_EXEC_GUARD
#include <nvbench/detail/state_exec.cuh>
#undef NVBENCH_STATE_EXEC_GUARD
