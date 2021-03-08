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

#include <nvbench/named_values.cuh>

#include <string>
#include <utility>

namespace nvbench
{

/**
 * A named set of key/value pairs associated with a benchmark result.
 *
 * The summary name is the unabbreviated name for the measurement.
 * An abbreviated name for column headings can be suggested in a "short_name"
 * entry (see below).
 *
 * Some keys have standard meanings that output formats may use to produce
 * more readable representations of the result:
 *
 * - "hint": Formatting hints (see below)
 * - "short_name": Abbreviated name for table headings.
 * - "description": Longer description of result.
 * - "value": Actual value.
 *
 * Hints:
 * - unset: Arbitrary value is stored in "value".
 * - "duration": "value" is a float64_t time duration in seconds.
 * - "item_rate": "value" is a float64_t item rate in elements / second.
 * - "bytes": "value" is an int64_t number of bytes.
 * - "byte_rate": "value" is a float64_t byte rate in bytes / second.
 * - "sample_size": "value" is an int64_t number of samples in a measurement.
 * - "percentage": "value" is a float64_t percentage.
 *
 * The key/value pair functionality is implemented by the
 * `nvbench::named_values` base class.
 *
 * Example: Adding a new summary to an nvbench::state object:
 *
 * ```
 * auto &summ = state.add_summary("Average GPU Time (Batch)");
 * summ.set_string("hint", "duration");
 * summ.set_string("short_name", "Batch GPU");
 * summ.set_string("description",
 *                 "Average back-to-back kernel execution time as measured "
 *                 "by CUDA events.");
 * summ.set_float64("value", avg_batch_gpu_time);
 * ```
 */
struct summary : public nvbench::named_values
{
  summary() = default;
  explicit summary(std::string name)
      : m_name(std::move(name))
  {}

  // move-only
  summary(const summary &) = delete;
  summary(summary &&)      = default;
  summary &operator=(const summary &) = delete;
  summary &operator=(summary &&) = default;

  void set_name(std::string name) { m_name = std::move(name); }
  [[nodiscard]] const std::string &get_name() const { return m_name; }

private:
  std::string m_name;
};

} // namespace nvbench
