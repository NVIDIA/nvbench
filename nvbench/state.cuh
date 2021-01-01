#pragma once

#include <nvbench/named_values.cuh>
#include <nvbench/types.cuh>

#include <string>

namespace nvbench
{

namespace detail
{
struct state_generator;
}

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

protected:
  friend struct nvbench::detail::state_generator;

  state() = default;

  state(nvbench::named_values values)
      : m_axis_values{std::move(values)}
  {}

  nvbench::named_values m_axis_values;
  std::string m_skip_reason;
};

} // namespace nvbench
