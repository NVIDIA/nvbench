#pragma once

#include <nvbench/named_values.cuh>
#include <nvbench/summary.cuh>
#include <nvbench/types.cuh>

#include <string>
#include <vector>

namespace nvbench
{

namespace detail
{
struct state_generator;
}

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

  [[nodiscard]] nvbench::int64_t get_int64(const std::string &axis_name) const;

  [[nodiscard]] nvbench::float64_t
  get_float64(const std::string &axis_name) const;

  [[nodiscard]] const std::string &
  get_string(const std::string &axis_name) const;

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

  summary &add_summary(std::string summary_name);
  summary &add_summary(summary s);
  [[nodiscard]] const summary &get_summary(std::string_view name) const;
  [[nodiscard]] summary &get_summary(std::string_view name);
  [[nodiscard]] const std::vector<summary> &get_summaries() const;
  [[nodiscard]] std::vector<summary> &get_summaries();

protected:
  friend struct nvbench::detail::state_generator;

  state() = default;

  state(nvbench::named_values values)
      : m_axis_values{std::move(values)}
  {}

  nvbench::named_values m_axis_values;
  std::vector<nvbench::summary> m_summaries;
  std::string m_skip_reason;
};

} // namespace nvbench
