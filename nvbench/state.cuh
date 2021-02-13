#pragma once

#include <nvbench/device_info.cuh>
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

  void set_items_processed_per_launch(nvbench::int64_t items)
  {
    m_items_processed_per_launch = items;
  }
  [[nodiscard]] nvbench::int64_t get_items_processed_per_launch() const
  {
    return m_items_processed_per_launch;
  }

  void set_global_bytes_accessed_per_launch(nvbench::int64_t bytes)
  {
    m_global_bytes_accessed_per_launch = bytes;
  }
  [[nodiscard]] nvbench::int64_t get_global_bytes_accessed_per_launch() const
  {
    return m_global_bytes_accessed_per_launch;
  }

  void skip(std::string reason) { m_skip_reason = std::move(reason); }
  [[nodiscard]] bool is_skipped() const { return !m_skip_reason.empty(); }
  [[nodiscard]] const std::string &get_skip_reason() const
  {
    return m_skip_reason;
  }

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

private:
  friend struct nvbench::detail::state_generator;
  friend struct nvbench::detail::state_tester;

  explicit state(const benchmark_base &bench)
      : m_benchmark{bench}
  {}

  state(const benchmark_base &bench,
        nvbench::named_values values,
        std::optional<nvbench::device_info> device,
        std::size_t type_config_index)
      : m_benchmark{bench}
      , m_axis_values{std::move(values)}
      , m_device{std::move(device)}
      , m_type_config_index{type_config_index}
  {}

  std::reference_wrapper<const nvbench::benchmark_base> m_benchmark;
  nvbench::named_values m_axis_values;
  std::optional<nvbench::device_info> m_device;
  std::size_t m_type_config_index{};

  std::vector<nvbench::summary> m_summaries;
  std::string m_skip_reason;
  nvbench::int64_t m_items_processed_per_launch{};
  nvbench::int64_t m_global_bytes_accessed_per_launch{};
};

} // namespace nvbench
