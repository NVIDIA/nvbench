#include <nvbench/state.cuh>

#include <nvbench/benchmark_base.cuh>
#include <nvbench/types.cuh>

#include <fmt/color.h>
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

std::string state::get_short_description() const
{
  fmt::memory_buffer buffer;

  fmt::format_to(
    buffer,
    "{}",
    fmt::format(fmt::emphasis::bold, "{}", m_benchmark.get().get_name()));

  buffer.push_back(' ');
  buffer.push_back('[');

  auto append_key_value = [&buffer](const std::string &key,
                                    const auto &value,
                                    std::string value_fmtstr = "{}") {
    constexpr auto key_format   = fmt::emphasis::italic;
    constexpr auto value_format = fmt::emphasis::bold;

    fmt::format_to(buffer,
                   "{}{}={}",
                   buffer.size() == 0 ? "" : " ",
                   fmt::format(key_format, "{}", key),
                   fmt::format(value_format, value_fmtstr, value));
  };

  if (m_device)
  {
    append_key_value("Device", m_device->get_id());
  }

  const axes_metadata &axes = m_benchmark.get().get_axes();
  for (const auto &name : m_axis_values.get_names())
  {
    // Handle power-of-two int64 axes differently:
    if (m_axis_values.get_type(name) == named_values::type::int64 &&
        axes.get_int64_axis(name).is_power_of_two())
    {
      const nvbench::uint64_t value    = m_axis_values.get_int64(name);
      const nvbench::uint64_t exponent = int64_axis::compute_log2(value);
      append_key_value(name, exponent, "2^{}");
    }
    else
    {
      auto visitor = [&name, &append_key_value](const auto &value) {
        append_key_value(name, value);
      };
      std::visit(visitor, m_axis_values.get_value(name));
    }
  }

  buffer.push_back(']');

  return fmt::to_string(buffer);
}

void state::add_element_count(std::size_t elements,
                              std::string column_name)
{
  m_element_count += static_cast<nvbench::int64_t>(elements);
  if (!column_name.empty())
  {
    auto &summ = this->add_summary("Element count: " + column_name);
    summ.set_string("short_name", std::move(column_name));
    summ.set_int64("value", static_cast<nvbench::int64_t>(elements));
  }
}

void state::add_global_memory_reads(std::size_t bytes, std::string column_name)
{
  m_global_memory_rw_bytes += static_cast<nvbench::int64_t>(bytes);
  if (!column_name.empty())
  {
    auto &summ = this->add_summary("Input Buffer Size: " + column_name);
    summ.set_string("hint", "bytes");
    summ.set_string("short_name", std::move(column_name));
    summ.set_int64("value", static_cast<nvbench::int64_t>(bytes));
  }
}

void state::add_global_memory_writes(std::size_t bytes, std::string column_name)
{
  m_global_memory_rw_bytes += static_cast<nvbench::int64_t>(bytes);
  if (!column_name.empty())
  {
    auto &summ = this->add_summary("Output Buffer Size: " + column_name);
    summ.set_string("hint", "bytes");
    summ.set_string("short_name", std::move(column_name));
    summ.set_int64("value", static_cast<nvbench::int64_t>(bytes));
  }
}

} // namespace nvbench
