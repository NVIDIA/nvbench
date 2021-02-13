#include <nvbench/detail/markdown_format.cuh>

#include <nvbench/benchmark_base.cuh>
#include <nvbench/device_manager.cuh>
#include <nvbench/state.cuh>
#include <nvbench/summary.cuh>

#include <nvbench/detail/transform_reduce.cuh>

#include <fmt/format.h>

#include <functional>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

namespace
{

struct table_builder
{
  void add_cell(std::size_t row,
                const std::string &column_key,
                const std::string &header,
                std::string value)
  {
    auto iter = std::find_if(m_columns.begin(),
                             m_columns.end(),
                             [&column_key](const column &col) {
                               return col.key == column_key;
                             });

    auto &col = iter == m_columns.end()
                  ? m_columns.emplace_back(column{column_key,
                                                  header,
                                                  std::vector<std::string>{},
                                                  header.size()})
                  : *iter;

    col.max_width = std::max(col.max_width, value.size());
    if (col.rows.size() <= row)
    {
      col.rows.resize(row + 1);
      col.rows[row] = std::move(value);
    }
  }

  std::string to_string()
  {
    fmt::memory_buffer buffer;
    this->fix_row_lengths();
    this->print_header(buffer);
    this->print_divider(buffer);
    this->print_rows(buffer);
    return fmt::to_string(buffer);
  }

private:
  struct column
  {
    std::string key;
    std::string header;
    std::vector<std::string> rows;
    std::size_t max_width;
  };

  void fix_row_lengths()
  { // Ensure that each row is the same length:
    m_num_rows = nvbench::detail::transform_reduce(
      m_columns.cbegin(),
      m_columns.cend(),
      0ll,
      [](const auto &a, const auto &b) { return a > b ? a : b; },
      [](const column &col) { return col.rows.size(); });
    std::for_each(m_columns.begin(),
                  m_columns.end(),
                  [num_rows = m_num_rows](column &col) {
                    col.rows.resize(num_rows);
                  });
  }

  void print_header(fmt::memory_buffer &buffer)
  {
    fmt::format_to(buffer, "|");
    for (const column &col : m_columns)
    {
      fmt::format_to(buffer, " {:^{}} |", col.header, col.max_width);
    }
    fmt::format_to(buffer, "\n");
  }

  void print_divider(fmt::memory_buffer &buffer)
  {
    fmt::format_to(buffer, "|");
    for (const column &col : m_columns)
    { // fill=-, centered, empty string, width = max_width + 2
      fmt::format_to(buffer, "{:-^{}}|", "", col.max_width + 2);
    }
    fmt::format_to(buffer, "\n");
  }

  void print_rows(fmt::memory_buffer &buffer)
  {
    for (std::size_t row = 0; row < m_num_rows; ++row)
    {
      fmt::format_to(buffer, "|");
      for (const column &col : m_columns)
      { // fill=-, centered, empty string, width = max_width + 2
        fmt::format_to(buffer, " {:>{}} |", col.rows[row], col.max_width);
      }
      fmt::format_to(buffer, "\n");
    }
  }

  std::vector<column> m_columns;
  std::size_t m_num_rows{};
};

} // namespace

namespace nvbench
{
namespace detail
{

void markdown_format::print_device_info()
{
  fmt::print("# Devices\n\n");

  const auto &devices = nvbench::device_manager::get().get_devices();
  for (const auto &device : devices)
  {
    const auto [gmem_free, gmem_used] = device.get_global_memory_usage();

    fmt::print("## [{}] `{}`\n", device.get_id(), device.get_name());
    fmt::print("* SM Version: {} (PTX Version: {})\n",
               device.get_sm_version(),
               device.get_ptx_version());
    fmt::print("* Number of SMs: {}\n", device.get_number_of_sms());
    fmt::print("* SM Default Clock Rate: {} MHz\n",
               device.get_sm_default_clock_rate() / 1000 / 1000);
    fmt::print("* Global Memory: {} MiB Free / {} MiB Total\n",
               gmem_free / 1024 / 1024,
               gmem_used / 1024 / 1024);
    fmt::print("* Global Memory Bus Peak: {} GiB/sec ({}-bit DDR @{}MHz)\n",
               device.get_global_memory_bus_bandwidth() / 1000 / 1000 / 1000,
               device.get_global_memory_bus_width(),
               device.get_global_memory_bus_peak_clock_rate() / 1000 / 1000);
    fmt::print("* Max Shared Memory: {} KiB/SM, {} KiB/Block\n",
               device.get_shared_memory_per_sm() / 1024,
               device.get_shared_memory_per_block() / 1024);
    fmt::print("* L2 Cache Size: {} KiB\n", device.get_l2_cache_size() / 1024);
    fmt::print("* Maximum Active Blocks: {}/SM\n",
               device.get_max_blocks_per_sm());
    fmt::print("* Maximum Active Threads: {}/SM, {}/Block\n",
               device.get_max_threads_per_sm(),
               device.get_max_threads_per_block());
    fmt::print("* Available Registers: {}/SM, {}/Block\n",
               device.get_registers_per_sm(),
               device.get_registers_per_block());
    fmt::print("* ECC Enabled: {}\n", device.get_ecc_state() ? "Yes" : "No");
    fmt::print("\n");
  }
}

void markdown_format::print_log_preamble() { fmt::print("# Log\n\n```\n"); }

void markdown_format::print_log_epilogue() { fmt::print("```\n\n"); }

void markdown_format::print_benchmark_summaries(
  const benchmark_vector &benchmarks)
{
  fmt::print("# Benchmarks\n\n");
  std::size_t benchmark_id{0};
  for (const auto &bench_ptr : benchmarks)
  {
    const auto &axes              = bench_ptr->get_axes().get_axes();
    const std::size_t num_configs = nvbench::detail::transform_reduce(
      axes.cbegin(),
      axes.cend(),
      std::size_t{1},
      std::multiplies<>{},
      [](const auto &axis_ptr) { return axis_ptr->get_size(); });

    fmt::print("## [{}] `{}` ({} configurations)\n\n",
               benchmark_id++,
               bench_ptr->get_name(),
               num_configs);

    fmt::print("### Axes\n\n");
    for (const auto &axis_ptr : axes)
    {
      std::string flags_str(axis_ptr->get_flags_as_string());
      if (!flags_str.empty())
      {
        flags_str = fmt::format(" [{}]", flags_str);
      }
      fmt::print("* `{}` : {}{}\n",
                 axis_ptr->get_name(),
                 axis_ptr->get_type_as_string(),
                 flags_str);

      const std::size_t num_vals = axis_ptr->get_size();
      for (std::size_t i = 0; i < num_vals; ++i)
      {
        std::string desc = axis_ptr->get_description(i);
        if (!desc.empty())
        {
          desc = fmt::format(" ({})", desc);
        }
        fmt::print("  * `{}`{}\n", axis_ptr->get_input_string(i), desc);
      } // end foreach value
    }   // end foreach axis
    fmt::print("\n");
  } // end foreach bench
}

void markdown_format::print_benchmark_results(const benchmark_vector &benchmarks)
{
  // This needs to be refactored and cleaned up (someday....) but here's a
  // buncha functors that do various string formatting stuff:
  auto format_visitor = [](const auto &v) {
    using T = std::decay_t<decltype(v)>;
    if constexpr (std::is_same_v<T, nvbench::float64_t>)
    {
      return fmt::format("{:.5g}", v);
    }
    else if constexpr (std::is_same_v<T, std::string>)
    {
      return v;
    }
    return fmt::format("{}", v);
  };

  auto format_duration = [](nvbench::float64_t seconds) {
    if (seconds >= 1.) // 1+ sec
    {
      return fmt::format("{:.2f} s", seconds);
    }
    else if (seconds >= 1e-2) // 10+ ms.
    {
      return fmt::format("{:.2f} ms", seconds * 1e3);
    }
    else if (seconds >= 1e-5) // 10+ us.
    {
      return fmt::format("{:.2f} us", seconds * 1e6);
    }
    else
    {
      return fmt::format("{:.2f} ns", seconds * 1e9);
    }
  };

  auto format_item_rate = [](nvbench::float64_t items_per_second) {
    if (items_per_second >= 1e9)
    {
      return fmt::format("{:0.2f} GHz", items_per_second * 1e-9);
    }
    else if (items_per_second >= 1e6)
    {
      return fmt::format("{:0.2f} MHz", items_per_second * 1e-6);
    }
    else if (items_per_second >= 1e3)
    {
      return fmt::format("{:0.2f} KHz", items_per_second * 1e-3);
    }
    else
    {
      return fmt::format("{:0.2f} Hz", items_per_second);
    }
  };

  auto format_bytes = [](nvbench::int64_t bytes) {
    if (bytes >= 10. * 1024. * 1024. * 1024.) // 10 GiB
    {
      return fmt::format("{:.2f} GiB", bytes / (1024. * 1024. * 1024.));
    }
    else if (bytes >= 10. * 1024. * 1024.) // 10 MiB
    {
      return fmt::format("{:.2f} MiB", bytes / (1024. * 1024.));
    }
    else if (bytes >= 10 * 1024) // 10 KiB.
    {
      return fmt::format("{:.2f} KiB", bytes / 1024.);
    }
    else
    {
      return fmt::format("{:.2f} B", static_cast<nvbench::float64_t>(bytes));
    }
  };

  auto format_byte_rate = [](nvbench::float64_t bytes_per_second) {
    if (bytes_per_second >= 10. * 1024. * 1024. * 1024.) // 10 GiB/s
    {
      return fmt::format("{:.2f} GiB/s",
                         bytes_per_second / (1024. * 1024. * 1024.));
    }
    else if (bytes_per_second >= 10. * 1024. * 1024.) // 10 MiB/s
    {
      return fmt::format("{:.2f} MiB/s", bytes_per_second / (1024. * 1024.));
    }
    else if (bytes_per_second >= 10. * 1024.) // 10 KiB/s.
    {
      return fmt::format("{:.2f} KiB/s", bytes_per_second / 1024.);
    }
    else
    {
      return fmt::format("{:.2f} B/s", bytes_per_second);
    }
  };

  auto format_percentage = [](nvbench::float64_t percentage) {
    return fmt::format("{:.2f}%", percentage);
  };

  // Start printing benchmarks
  fmt::print("# Benchmark Results\n");

  for (const auto &bench_ptr : benchmarks)
  {
    const auto &bench   = *bench_ptr;
    const auto &devices = bench.get_devices();
    const auto &axes    = bench.get_axes();

    fmt::print("\n## {}\n", bench.get_name());

    // Do a single pass when no devices are specified. This happens for
    // benchmarks with `cpu` exec_tags.
    const std::size_t num_device_passes = devices.empty() ? 1 : devices.size();
    for (std::size_t device_pass = 0; device_pass < num_device_passes;
         ++device_pass)
    {
      std::optional<nvbench::device_info> device =
        devices.empty() ? std::nullopt
                        : std::make_optional(devices[device_pass]);

      if (device)
      {
        fmt::print("\n### [{}] {}\n\n", device->get_id(), device->get_name());
      }

      std::size_t row = 0;
      table_builder table;

      for (const auto &cur_state : bench.get_states())
      {
        if (cur_state.get_device() == device)
        {
          const auto &axis_values = cur_state.get_axis_values();
          for (const auto &name : axis_values.get_names())
          {
            // Handle power-of-two int64 axes differently:
            if (axis_values.get_type(name) == named_values::type::int64 &&
                axes.get_int64_axis(name).is_power_of_two())
            {
              const nvbench::int64_t value    = axis_values.get_int64(name);
              const nvbench::int64_t exponent = int64_axis::compute_log2(value);
              table.add_cell(row,
                             name + "_axis_pretty",
                             name,
                             fmt::format("2^{}", exponent));
              table.add_cell(row,
                             name + "_axis_descriptive",
                             fmt::format("({})", name),
                             fmt::to_string(value));
            }
            else
            {
              std::string value = std::visit(format_visitor,
                                             axis_values.get_value(name));
              table.add_cell(row, name + "_axis", name, std::move(value));
            }
          }

          for (const auto &summ : cur_state.get_summaries())
          {
            if (summ.has_value("hide"))
            {
              continue;
            }
            const std::string &key    = summ.get_name();
            const std::string &header = summ.has_value("short_name")
                                          ? summ.get_string("short_name")
                                          : key;

            std::string hint = summ.has_value("hint") ? summ.get_string("hint")
                                                      : std::string{};
            if (hint == "duration")
            {
              table.add_cell(row,
                             key,
                             header,
                             format_duration(summ.get_float64("value")));
            }
            else if (hint == "item_rate")
            {
              table.add_cell(row,
                             key,
                             header,
                             format_item_rate(summ.get_float64("value")));
            }
            else if (hint == "bytes")
            {
              table.add_cell(row,
                             key,
                             header,
                             format_bytes(summ.get_int64("value")));
            }
            else if (hint == "byte_rate")
            {
              table.add_cell(row,
                             key,
                             header,
                             format_byte_rate(summ.get_float64("value")));
            }
            else if (hint == "percentage")
            {
              table.add_cell(row,
                             key,
                             header,
                             format_percentage(summ.get_float64("value")));
            }
            else
            {
              table.add_cell(row,
                             key,
                             header,
                             std::visit(format_visitor,
                                        summ.get_value("value")));
            }
          }
          row++;
        }
      }

      fmt::print("{}", table.to_string());
    } // end foreach device_pass
  }
}

} // namespace detail
} // namespace nvbench
