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

protected:
  friend struct nvbench::detail::state_generator;

  state() = default;

  nvbench::named_values m_axis_values;
};

} // namespace nvbench
