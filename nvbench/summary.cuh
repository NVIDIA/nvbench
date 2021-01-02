#pragma once

#include <nvbench/named_values.cuh>

#include <string>
#include <utility>

namespace nvbench
{

/**
 * A named set of key/value pairs associated with a measurement.
 *
 * The key/value pair functionality is implemented by the `named_values` base
 * class.
 *
 * Some keys have standard meanings that output formats may use to produce
 * better representations of the summary.
 * @todo TODO fill this out as the format writers develop. These are some ideas:
 * - "hint": {"duration", "bandwidth", "bytes", "etc}
 * - "fmt_string": "{:9.5f}"
 * - "short_name": "%PeakMBW" (Abbreviated name for table headings)
 * - "description": "Average global device memory throughput as a percentage of the device's peak bandwidth."
 *
 * Hints:
 * - "hint" unset: Arbitrary value is stored in a key named "value".
 * - "hint" == "duration":
 *   - "value" is a float64_t with the mean elapsed time in seconds.
 *   - Additional optional float64_t keys: "min", "max", "stdev"
 */
struct summary : public nvbench::named_values
{
  summary() = default;
  explicit summary(std::string name)
      : m_name(std::move(name))
  {}

  void set_name(std::string name) { m_name = std::move(name); }
  const std::string &get_name() const { return m_name; }

private:
  std::string m_name;
};

} // namespace nvbench
