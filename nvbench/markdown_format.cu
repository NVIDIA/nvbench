#include <nvbench/markdown_format.cuh>

#include <nvbench/benchmark_base.cuh>
#include <nvbench/device_manager.cuh>
#include <nvbench/state.cuh>
#include <nvbench/summary.cuh>

#include <nvbench/detail/transform_reduce.cuh>

#include <nvbench/internal/markdown_table.cuh>

#include <fmt/format.h>

#include <functional>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

namespace nvbench
{

void markdown_format::do_print_device_info()
{
  fmt::memory_buffer buffer;
  fmt::format_to(buffer, "# Devices\n\n");

  const auto &devices = nvbench::device_manager::get().get_devices();
  for (const auto &device : devices)
  {
    const auto [gmem_free, gmem_used] = device.get_global_memory_usage();

    fmt::format_to(buffer, "## [{}] `{}`\n", device.get_id(), device.get_name());
    fmt::format_to(buffer,
                   "* SM Version: {} (PTX Version: {})\n",
                   device.get_sm_version(),
                   device.get_ptx_version());
    fmt::format_to(buffer, "* Number of SMs: {}\n", device.get_number_of_sms());
    fmt::format_to(buffer,
                   "* SM Default Clock Rate: {} MHz\n",
                   device.get_sm_default_clock_rate() / 1000 / 1000);
    fmt::format_to(buffer,
                   "* Global Memory: {} MiB Free / {} MiB Total\n",
                   gmem_free / 1024 / 1024,
                   gmem_used / 1024 / 1024);
    fmt::format_to(
      buffer,
      "* Global Memory Bus Peak: {} GiB/sec ({}-bit DDR @{}MHz)\n",
      device.get_global_memory_bus_bandwidth() / 1000 / 1000 / 1000,
      device.get_global_memory_bus_width(),
      device.get_global_memory_bus_peak_clock_rate() / 1000 / 1000);
    fmt::format_to(buffer,
                   "* Max Shared Memory: {} KiB/SM, {} KiB/Block\n",
                   device.get_shared_memory_per_sm() / 1024,
                   device.get_shared_memory_per_block() / 1024);
    fmt::format_to(buffer,
                   "* L2 Cache Size: {} KiB\n",
                   device.get_l2_cache_size() / 1024);
    fmt::format_to(buffer,
                   "* Maximum Active Blocks: {}/SM\n",
                   device.get_max_blocks_per_sm());
    fmt::format_to(buffer,
                   "* Maximum Active Threads: {}/SM, {}/Block\n",
                   device.get_max_threads_per_sm(),
                   device.get_max_threads_per_block());
    fmt::format_to(buffer,
                   "* Available Registers: {}/SM, {}/Block\n",
                   device.get_registers_per_sm(),
                   device.get_registers_per_block());
    fmt::format_to(buffer,
                   "* ECC Enabled: {}\n",
                   device.get_ecc_state() ? "Yes" : "No");
    fmt::format_to(buffer, "\n");
  }
  m_ostream << fmt::to_string(buffer);
}

void markdown_format::do_print_log_preamble() { m_ostream << "# Log\n\n```\n"; }

void markdown_format::do_print_log_epilogue() { m_ostream << "```\n\n"; }

void markdown_format::do_print_benchmark_list(
  const output_format::benchmark_vector &benches)
{
  fmt::memory_buffer buffer;
  fmt::format_to(buffer, "# Benchmarks\n\n");
  std::size_t benchmark_id{0};
  for (const auto &bench_ptr : benches)
  {
    const auto &axes              = bench_ptr->get_axes().get_axes();
    const std::size_t num_configs = nvbench::detail::transform_reduce(
      axes.cbegin(),
      axes.cend(),
      std::size_t{1},
      std::multiplies<>{},
      [](const auto &axis_ptr) { return axis_ptr->get_size(); });

    fmt::format_to(buffer,
                   "## [{}] `{}` ({} configurations)\n\n",
                   benchmark_id++,
                   bench_ptr->get_name(),
                   num_configs);

    fmt::format_to(buffer, "### Axes\n\n");
    for (const auto &axis_ptr : axes)
    {
      std::string flags_str(axis_ptr->get_flags_as_string());
      if (!flags_str.empty())
      {
        flags_str = fmt::format(" [{}]", flags_str);
      }
      fmt::format_to(buffer,
                     "* `{}` : {}{}\n",
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
        fmt::format_to(buffer,
                       "  * `{}`{}\n",
                       axis_ptr->get_input_string(i),
                       desc);
      } // end foreach value
    }   // end foreach axis
    fmt::format_to(buffer, "\n");
  } // end foreach bench

  m_ostream << fmt::to_string(buffer);
}

void markdown_format::do_print_benchmark_results(
  const output_format::benchmark_vector &benches)
{
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

  // Start printing benchmarks
  fmt::memory_buffer buffer;
  fmt::format_to(buffer, "# Benchmark Results\n");

  for (const auto &bench_ptr : benches)
  {
    const auto &bench   = *bench_ptr;
    const auto &devices = bench.get_devices();
    const auto &axes    = bench.get_axes();

    fmt::format_to(buffer, "\n## {}\n", bench.get_name());

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
        fmt::format_to(buffer,
                       "\n### [{}] {}\n\n",
                       device->get_id(),
                       device->get_name());
      }

      std::size_t row = 0;
      nvbench::internal::markdown_table table;

      for (const auto &cur_state : bench.get_states())
      {
        if (cur_state.is_skipped())
        {
          continue;
        }

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
              table.add_cell(row, key, header, this->do_format_duration(summ));
            }
            else if (hint == "item_rate")
            {
              table.add_cell(row, key, header, this->do_format_item_rate(summ));
            }
            else if (hint == "bytes")
            {
              table.add_cell(row, key, header, this->do_format_bytes(summ));
            }
            else if (hint == "byte_rate")
            {
              table.add_cell(row, key, header, this->do_format_byte_rate(summ));
            }
            else if (hint == "sample_size")
            {
              table.add_cell(row,
                             key,
                             header,
                             this->do_format_sample_size(summ));
            }
            else if (hint == "percentage")
            {
              table.add_cell(row, key, header, this->do_format_percentage(summ));
            }
            else
            {
              table.add_cell(row, key, header, this->do_format_default(summ));
            }
          }
          row++;
        }
      }

      auto table_str = table.to_string();
      fmt::format_to(buffer,
                     "{}",
                     table_str.empty() ? "No data -- check log.\n"
                                       : std::move(table_str));
    } // end foreach device_pass
  }

  m_ostream << fmt::to_string(buffer);
}

std::string markdown_format::do_format_default(const summary &data)
{
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

  return std::visit(format_visitor, data.get_value("value"));
}

std::string markdown_format::do_format_duration(const summary &data)
{
  const auto seconds = data.get_float64("value");
  if (seconds >= 1.) // 1+ sec
  {
    return fmt::format("{:0.3f} s", seconds);
  }
  else if (seconds >= 1e-3) // 1+ ms.
  {
    return fmt::format("{:0.3f} ms", seconds * 1e3);
  }
  else if (seconds >= 1e-6) // 1+ us.
  {
    return fmt::format("{:0.3f} us", seconds * 1e6);
  }
  else
  {
    return fmt::format("{:0.3f} ns", seconds * 1e9);
  }
}

std::string markdown_format::do_format_item_rate(const summary &data)
{
  const auto items_per_second = data.get_float64("value");
  if (items_per_second >= 1e9)
  {
    return fmt::format("{:0.3f}b", items_per_second * 1e-9);
  }
  else if (items_per_second >= 1e6)
  {
    return fmt::format("{:0.3f}m", items_per_second * 1e-6);
  }
  else if (items_per_second >= 1e3)
  {
    return fmt::format("{:0.3f}k", items_per_second * 1e-3);
  }
  else
  {
    return fmt::format("{:0.3f}", items_per_second);
  }
}

std::string markdown_format::do_format_bytes(const summary &data)
{
  const auto bytes = data.get_int64("value");
  if (bytes >= 1024. * 1024. * 1024.) // 1 GiB
  {
    return fmt::format("{:0.3f} GiB", bytes / (1024. * 1024. * 1024.));
  }
  else if (bytes >= 1024. * 1024.) // 1 MiB
  {
    return fmt::format("{:0.3f} MiB", bytes / (1024. * 1024.));
  }
  else if (bytes >= 1024) // 1 KiB.
  {
    return fmt::format("{:0.3f} KiB", bytes / 1024.);
  }
  else
  {
    return fmt::format("{:0.3f} B", static_cast<nvbench::float64_t>(bytes));
  }
}

std::string markdown_format::do_format_byte_rate(const summary &data)
{
  const auto bytes_per_second = data.get_float64("value");
  if (bytes_per_second >= 1024. * 1024. * 1024.) // 1 GiB/s
  {
    return fmt::format("{:0.3f} GiB/s",
                       bytes_per_second / (1024. * 1024. * 1024.));
  }
  else if (bytes_per_second >= 1024. * 1024.) // 1 MiB/s
  {
    return fmt::format("{:0.3f} MiB/s", bytes_per_second / (1024. * 1024.));
  }
  else if (bytes_per_second >= 1024.) // 1 KiB/s.
  {
    return fmt::format("{:0.3f} KiB/s", bytes_per_second / 1024.);
  }
  else
  {
    return fmt::format("{:0.3f} B/s", bytes_per_second);
  }
}

std::string markdown_format::do_format_sample_size(const summary &data)
{
  const auto count = data.get_int64("value");
  return fmt::format("{}x", count);
}

std::string markdown_format::do_format_percentage(const summary &data)
{
  const auto percentage = data.get_float64("value");
  return fmt::format("{:.2f}%", percentage);
}

} // namespace nvbench
