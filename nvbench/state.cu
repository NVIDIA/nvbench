#include <nvbench/state.cuh>

#include <nvbench/benchmark_base.cuh>
#include <nvbench/types.cuh>

#include <fmt/format.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace nvbench
{

state::state(const benchmark_base &bench)
    : m_benchmark{bench}
    , m_min_samples{bench.get_min_samples()}
    , m_min_time{bench.get_min_time()}
    , m_max_noise{bench.get_max_noise()}
    , m_skip_time{bench.get_skip_time()}
    , m_timeout{bench.get_timeout()}
{}

state::state(const benchmark_base &bench,
             nvbench::named_values values,
             std::optional<nvbench::device_info> device,
             std::size_t type_config_index)
    : m_benchmark{bench}
    , m_axis_values{std::move(values)}
    , m_device{std::move(device)}
    , m_type_config_index{type_config_index}
    , m_min_samples{bench.get_min_samples()}
    , m_min_time{bench.get_min_time()}
    , m_max_noise{bench.get_max_noise()}
    , m_skip_time{bench.get_skip_time()}
    , m_timeout{bench.get_timeout()}
{}

nvbench::int64_t state::get_int64(const std::string &axis_name) const
{
  return m_axis_values.get_int64(axis_name);
}

nvbench::float64_t state::get_float64(const std::string &axis_name) const
{
  return m_axis_values.get_float64(axis_name);
}

const std::string &state::get_string(const std::string &axis_name) const
{
  return m_axis_values.get_string(axis_name);
}

summary &state::add_summary(std::string summary_name)
{
  return m_summaries.emplace_back(std::move(summary_name));
}

summary &state::add_summary(summary s)
{
  m_summaries.push_back(std::move(s));
  return m_summaries.back();
}

const summary &state::get_summary(std::string_view name) const
{
  auto iter =
    std::find_if(m_summaries.cbegin(),
                 m_summaries.cend(),
                 [&name](const auto &s) { return s.get_name() == name; });
  if (iter == m_summaries.cend())
  {
    throw std::runtime_error(
      fmt::format("{}:{}: No summary named '{}'.", __FILE__, __LINE__, name));
  }
  return *iter;
}

summary &state::get_summary(std::string_view name)
{
  auto iter = std::find_if(m_summaries.begin(),
                           m_summaries.end(),
                           [&name](auto &s) { return s.get_name() == name; });
  if (iter == m_summaries.end())
  {
    throw std::runtime_error(
      fmt::format("{}:{}: No summary named '{}'.", __FILE__, __LINE__, name));
  }
  return *iter;
}

const std::vector<summary> &state::get_summaries() const { return m_summaries; }

std::vector<summary> &state::get_summaries() { return m_summaries; }

} // namespace nvbench
