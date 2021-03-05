/*
 *  Copyright 2020 NVIDIA Corporation
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
#include <nvbench/device_info.cuh>
#include <nvbench/device_manager.cuh>
#include <nvbench/summary.cuh>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

namespace
{

template <typename JsonNode>
void write_named_values(JsonNode &node, const nvbench::named_values &values)
{
  const auto value_names = values.get_names();
  for (const auto &value_name : value_names)
  {
    const auto value_index = node.size();
    auto &value            = node[value_index];

    value["name"] = value_name;

    const auto type = values.get_type(value_name);
    switch (type)
    {
      case nvbench::named_values::type::int64:
        value["type"]  = "int64";
        value["value"] = values.get_int64(value_name);
        break;

      case nvbench::named_values::type::float64:
        value["type"]  = "float64";
        value["value"] = values.get_float64(value_name);
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
        const auto axis_index = axes.size();
        auto &axis            = axes[axis_index];

        axis["index"] = axis_index;
        axis["name"]  = axis_ptr->get_name();
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
        const auto state_index = states.size();
        auto &st               = states[state_index];

        st["index"]             = state_index;
        st["description"]       = exec_state.get_short_description();
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
          const auto summ_index = summaries.size();
          auto &summ            = summaries[summ_index];

          summ["index"] = summ_index;
          summ["name"]  = exec_summ.get_name();

          ::write_named_values(summ["values"], exec_summ);
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
