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

#pragma once

#include <nvbench/printer_base.cuh>

#include <nvbench/types.cuh>

namespace nvbench
{

/*!
 * JSON output format.
 */
struct json_printer : nvbench::printer_base
{
  using printer_base::printer_base;

  json_printer(std::ostream &stream,
               std::string stream_name,
               bool enable_binary_output)
      : printer_base(stream, std::move(stream_name))
      , m_enable_binary_output{enable_binary_output}
  {}

  [[nodiscard]] bool get_enable_binary_output() const
  {
    return m_enable_binary_output;
  }
  void set_enable_binary_output(bool b) { m_enable_binary_output = b; }

protected:
  // Virtual API from printer_base:
  void do_process_bulk_data_float64(
    nvbench::state &state,
    const std::string &tag,
    const std::string &hint,
    const std::vector<nvbench::float64_t> &data) override;
  void do_print_benchmark_results(const benchmark_vector &benches) override;

  bool m_enable_binary_output{false};
  std::size_t m_num_jsonbin_files{};
};

} // namespace nvbench
