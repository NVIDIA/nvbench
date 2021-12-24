/*
 *  Copyright 2021 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 with the LLVM exception
 *  (the "License"); you may not use this file except in compliance with
 *  the License.
 *
 *  You may obtain a copy of the License at
 *
 *      http://llvm.org/foundation/relicensing/LICENSE.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <nvbench/json_printer.cuh>

#include <nvbench/axes_metadata.cuh>
#include <nvbench/benchmark_base.cuh>
#include <nvbench/config.cuh>
#include <nvbench/device_info.cuh>
#include <nvbench/device_manager.cuh>
#include <nvbench/state.cuh>
#include <nvbench/summary.cuh>

#include <nvbench/detail/throw.cuh>

#include <fmt/format.h>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if NVBENCH_CPP_DIALECT >= 2020
#include <bit>
#endif

namespace
{

bool is_little_endian()
{
#if NVBENCH_CPP_DIALECT >= 2020
  return std::endian::native == std::endian::little;
#else
  const nvbench::uint32_t word = {0xBadDecaf};
  nvbench::uint8_t bytes[4];
  std::memcpy(bytes, &word, 4);
  return bytes[0] == 0xaf;
#endif
}

template <typename JsonNode>
void write_named_values(JsonNode &node, const nvbench::named_values &values)
{
  const auto value_names = values.get_names();
  for (const auto &value_name : value_names)
  {
    auto &value = node[value_name];

    const auto type = values.get_type(value_name);
    switch (type)
    {
      case nvbench::named_values::type::int64:
        value["type"] = "int64";
        // Write as a string; JSON encodes all numbers as double-precision
        // floats, which would truncate int64s.
        value["value"] = fmt::to_string(values.get_int64(value_name));
        break;

      case nvbench::named_values::type::float64:
        value["type"] = "float64";
        // Write as a string for consistency with int64.
        value["value"] = fmt::to_string(values.get_float64(value_name));
        break;

      case nvbench::named_values::type::string:
        value["type"]  = "string";
        value["value"] = values.get_string(value_name);
        break;
    } // end switch (value type)
  }   // end foreach value name
}

} // end namespace

namespace nvbench
{

void json_printer::do_process_bulk_data_float64(
  state &state,
  const std::string &tag,
  const std::string &hint,
  const std::vector<nvbench::float64_t> &data)
{
  printer_base::do_process_bulk_data_float64(state, tag, hint, data);

  if (!m_enable_binary_output)
  {
    return;
  }

  if (hint == "sample_times")
  {
    namespace fs = std::filesystem;

    nvbench::cpu_timer timer;
    timer.start();

    fs::path result_path{m_stream_name + "-bin/"};
    try
    {
      if (!fs::exists(result_path))
      {
        if (!fs::create_directory(result_path))
        {
          NVBENCH_THROW(std::runtime_error,
                        "{}",
                        "Failed to create result directory '{}'.");
        }
      }
      else if (!fs::is_directory(result_path))
      {
        NVBENCH_THROW(std::runtime_error,
                      "{}",
                      "'{}' exists and is not a directory.");
      }

      const auto file_id = m_num_jsonbin_files++;
      result_path /= fmt::format("{:d}.bin", file_id);

      std::ofstream out;
      out.exceptions(out.exceptions() | std::ios::failbit | std::ios::badbit);
      out.open(result_path, std::ios::binary | std::ios::out);

      // FIXME: SLOW -- Writing the binary file, 4 bytes at a time...
      // There are a lot of optimizations that could be done here if this ends
      // up being a noticeable bottleneck.
      for (auto value64 : data)
      {
        const auto value32 = static_cast<nvbench::float32_t>(value64);
        char buffer[4];
        std::memcpy(buffer, &value32, 4);
        // the c++17 implementation of is_little_endian isn't constexpr, but
        // all supported compilers optimize this branch as if it were.
        if (!is_little_endian())
        {
          using std::swap;
          swap(buffer[0], buffer[3]);
          swap(buffer[1], buffer[2]);
        }
        out.write(buffer, 4);
      }
    }
    catch (std::exception &e)
    {
      if (auto printer_opt_ref = state.get_benchmark().get_printer();
          printer_opt_ref.has_value())
      {
        auto &printer = printer_opt_ref.value().get();
        printer.log(nvbench::log_level::warn,
                    fmt::format("Error writing {} ({}) to {}: {}",
                                tag,
                                hint,
                                result_path.string(),
                                e.what()));
      }
    } // end catch

    auto &summ = state.add_summary(fmt::format("nv/json/bin:{}", tag));
    summ.set_string("name", "Samples Times File");
    summ.set_string("hint", "file/sample_times");
    summ.set_string("description",
                    "Binary file containing sample times as little-endian "
                    "float32.");
    summ.set_string("filename", result_path.string());
    summ.set_int64("size", static_cast<nvbench::int64_t>(data.size()));
    summ.set_string("hide", "Not needed in table.");

    timer.stop();
    if (auto printer_opt_ref = state.get_benchmark().get_printer();
        printer_opt_ref.has_value())
    {
      auto &printer = printer_opt_ref.value().get();
      printer.log(nvbench::log_level::info,
                  fmt::format("Wrote '{}' in {:>6.3f}ms",
                              result_path.string(),
                              timer.get_duration() * 1000));
    }
  } // end hint == sample_times
}

void json_printer::do_print_benchmark_results(const benchmark_vector &benches)
{
  nlohmann::ordered_json root;

  {
    auto &devices = root["devices"];
    for (const auto &dev_info : nvbench::device_manager::get().get_devices())
    {
      auto &device                    = devices[devices.size()];
      device["id"]                    = dev_info.get_id();
      device["name"]                  = dev_info.get_name();
      device["sm_version"]            = dev_info.get_sm_version();
      device["ptx_version"]           = dev_info.get_ptx_version();
      device["sm_default_clock_rate"] = dev_info.get_sm_default_clock_rate();
      device["number_of_sms"]         = dev_info.get_number_of_sms();
      device["max_blocks_per_sm"]     = dev_info.get_max_blocks_per_sm();
      device["max_threads_per_sm"]    = dev_info.get_max_threads_per_sm();
      device["max_threads_per_block"] = dev_info.get_max_threads_per_block();
      device["registers_per_sm"]      = dev_info.get_registers_per_sm();
      device["registers_per_block"]   = dev_info.get_registers_per_block();
      device["global_memory_size"]    = dev_info.get_global_memory_size();
      device["global_memory_bus_peak_clock_rate"] =
        dev_info.get_global_memory_bus_peak_clock_rate();
      device["global_memory_bus_width"] =
        dev_info.get_global_memory_bus_width();
      device["global_memory_bus_bandwidth"] =
        dev_info.get_global_memory_bus_bandwidth();
      device["l2_cache_size"]        = dev_info.get_l2_cache_size();
      device["shared_memory_per_sm"] = dev_info.get_shared_memory_per_sm();
      device["shared_memory_per_block"] =
        dev_info.get_shared_memory_per_block();
      device["ecc_state"] = dev_info.get_ecc_state();
    }
  }

  {
    auto &benchmarks = root["benchmarks"];
    for (const auto &bench_ptr : benches)
    {
      const auto bench_index = benchmarks.size();
      auto &bench            = benchmarks[bench_index];

      bench["index"] = bench_index;
      bench["name"]  = bench_ptr->get_name();

      bench["min_samples"] = bench_ptr->get_min_samples();
      bench["min_time"]    = bench_ptr->get_min_time();
      bench["max_noise"]   = bench_ptr->get_max_noise();
      bench["skip_time"]   = bench_ptr->get_skip_time();
      bench["timeout"]     = bench_ptr->get_timeout();

      auto &devices = bench["devices"];
      for (const auto &dev_info : bench_ptr->get_devices())
      {
        devices.push_back(dev_info.get_id());
      }

      auto &axes = bench["axes"];
      for (const auto &axis_ptr : bench_ptr->get_axes().get_axes())
      {
        auto &axis = axes[axis_ptr->get_name()];

        axis["type"]  = axis_ptr->get_type_as_string();
        axis["flags"] = axis_ptr->get_flags_as_string();

        auto &values         = axis["values"];
        const auto axis_size = axis_ptr->get_size();
        for (std::size_t i = 0; i < axis_size; ++i)
        {
          const auto value_idx  = values.size();
          auto &value           = values[value_idx];
          value["input_string"] = axis_ptr->get_input_string(i);
          value["description"]  = axis_ptr->get_description(i);

          switch (axis_ptr->get_type())
          {
            case nvbench::axis_type::type:
              value["is_active"] =
                static_cast<type_axis &>(*axis_ptr).get_is_active(i);
              break;

            case nvbench::axis_type::int64:
              value["value"] =
                static_cast<int64_axis &>(*axis_ptr).get_value(i);
              break;

            case nvbench::axis_type::float64:
              value["value"] =
                static_cast<float64_axis &>(*axis_ptr).get_value(i);
              break;

            case nvbench::axis_type::string:
              value["value"] =
                static_cast<string_axis &>(*axis_ptr).get_value(i);
              break;
            default:
              break;
          } // end switch (axis type)
        }   // end foreach axis value
      }     // end foreach axis

      auto &states = bench["states"];
      for (const auto &exec_state : bench_ptr->get_states())
      {
        auto &st = states[exec_state.get_axis_values_as_string()];

        // TODO: Determine if these need to be part of the state key as well
        // for uniqueness. The device already is, but the type config index is
        // not.
        st["device"]            = exec_state.get_device()->get_id();
        st["type_config_index"] = exec_state.get_type_config_index();

        st["min_samples"] = exec_state.get_min_samples();
        st["min_time"]    = exec_state.get_min_time();
        st["max_noise"]   = exec_state.get_max_noise();
        st["skip_time"]   = exec_state.get_skip_time();
        st["timeout"]     = exec_state.get_timeout();

        ::write_named_values(st["axis_values"], exec_state.get_axis_values());

        auto &summaries = st["summaries"];
        for (const auto &exec_summ : exec_state.get_summaries())
        {
          auto &summ = summaries[exec_summ.get_tag()];
          ::write_named_values(summ, exec_summ);
        }

        st["is_skipped"] = exec_state.is_skipped();
        if (exec_state.is_skipped())
        {
          st["skip_reason"] = exec_state.get_skip_reason();
          continue;
        }
      } // end foreach exec_state
    }   // end foreach benchmark
  }

  m_ostream << root.dump(2) << "\n";
}

} // namespace nvbench
