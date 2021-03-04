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

#include <string>

// TODO These should exactly match the relevant files in /docs.
// Eventually we should generate this file at configure-time with CMake magic.

namespace nvbench::internal
{

static const std::string help_text =
  R"string_bounds(# Queries

* `--list`, `-l`
  * List all devices and benchmarks without running them.

* `--help`, `-h`
  * Print usage information and exit.

* `--help-axes`, `--help-axis`
  * Print axis specification documentation and exit.

# Output

* `--csv <filename/stream>`
  * Write CSV output to a file, or "stdout" / "stderr".

* `--markdown <filename/stream>`, `--md <filename/stream>`
  * Write markdown output to a file, or "stdout" / "stderr".
  * Markdown is written to "stdout" by default.

* `--quiet`, `-q`
  * Suppress output.

* `--color`
  * Use color in output (markdown + stdout only).

# Benchmark / Axis Specification

* `--benchmark <benchmark name/index>`, `-b <benchmark name/index>`
  * Execute a specific benchmark.
  * Argument is a benchmark name or index, taken from `--list`.
  * If not specified, all benchmarks will run.
  * `--benchmark` may be specified multiple times to run several benchmarks.
  * The same benchmark may be specified multiple times with different
    configurations.

* `--axis <axis specification>`, `-a <axis specification>`
  * Override an axis specification.
  * See `--help-axis` for details on axis specifications.
  * Applies to the most recent `--benchmark`, or all benchmarks if specified
    before any `--benchmark` arguments.

# Benchmark Properties

* `--devices <device ids>`, `--device <device ids>`, `-d <device ids>`
  * Limit execution to one or more devices.
  * `<device ids>` is a single id, or a comma separated list.
  * Device ids can be obtained from `--list`.
  * Applies to the most recent `--benchmark`, or all benchmarks if specified
    before any `--benchmark` arguments.

* `--min-samples <count>`
  * Gather at least `<count>` samples per measurement.
  * Default is 10 samples.
  * Applies to the most recent `--benchmark`, or all benchmarks if specified
    before any `--benchmark` arguments.

* `--min-time <seconds>`
  * Accumulate at least `<seconds>` of execution time per measurement.
  * Default is 0.5 seconds.
  * If both GPU and CPU times are gathered, this applies to GPU time only.
  * Applies to the most recent `--benchmark`, or all benchmarks if specified
    before any `--benchmark` arguments.

* `--max-noise <value>`
  * Gather samples until the error in the measurement drops below `<value>`.
  * Noise is computed as the percent relative standard deviation.
  * Default is 0.5%.
  * Only applies to Cold measurements.
  * If both GPU and CPU times are gathered, this applies to GPU noise only.
  * Applies to the most recent `--benchmark`, or all benchmarks if specified
    before any `--benchmark` arguments.

* `--skip-time <seconds>`
  * Skip a measurement when a warmup run executes in less than `<seconds>`.
  * Default is -1 seconds (disabled).
  * Intended for testing / debugging only.
  * Very fast kernels (<5us) often require an extremely large number of samples
    to converge `max-noise`. This option allows them to be skipped to save time
    during testing.
  * Applies to the most recent `--benchmark`, or all benchmarks if specified
    before any `--benchmark` arguments.

* `--timeout <seconds>`
  * Measurements will timeout after `<seconds>` have elapsed.
  * Default is 15 seconds.
  * `<seconds>` is walltime, not accumulated sample time.
  * If a measurement times out, the default markdown log will print a warning to
    report any outstanding termination criteria (min samples, min time, max
    noise).
  * Applies to the most recent `--benchmark`, or all benchmarks if specified
    before any `--benchmark` arguments.
)string_bounds";

static const std::string help_axis_text =
  R"string_bounds(# Axis Specification

The `--axis <axis spec>` option redefines the values in a benchmark's axis. It
applies to the benchmark created by the most recent `--benchmark` argument, or
all benchmarks if it precedes all `--benchmark` arguments (if any).

Valid axis specification follow the form:

* `<axis_name>=<value>`
* `<axis_name>=[<value1>,<value2>,...]`
* `<axis_name>=[<start>:<stop>]`
* `<axis_name>=[<start>:<stop>:<stride>]`
* `<axis_name>[<flags>]=<value>`
* `<axis_name>[<flags>]=[<value1>,<value2>,...]`
* `<axis_name>[<flags>]=[<start>:<stop>]`
* `<axis_name>[<flags>]=[<start>:<stop>:<stride>]`

Whitespace is ignored if the argument is quoted.

The axis type is taken from the benchmark definition. Some axes have additional
restrictions:

* Numeric axes:
  * A single value, explicit list of values, or strided range may be specified.
  * For `int64` axes, the `power_of_two` flag is specified by adding `[pow2]`
    after the axis name.
  * Values may differ from those defined in the benchmark.
* String axes:
  * A single value or explicit list of values may be specified.
  * Values may differ from those defined in the benchmark.
* Type axes:
  * A single value or explicit list of values may be specified.
  * Values **MUST** be a subset of the types defined in the benchmark.
  * Values **MUST** match the input strings provided by `--list` (e.g. `I32`
    for `int`).
  * Provide a `nvbench::type_strings<T>` specialization to modify a custom
    type's input string.

# Examples

## Single Value

| Axis Type | Example                 | Example Result   |
|-----------|-------------------------|------------------|
| Int64     | `-a InputSize=12345`    | 12345            |
| Int64Pow2 | `-a InputSize[pow2]=8`  | 256              |
| Float64   | `-a Quality=0.5`        | 0.5              |
| String    | `-a RNG=Uniform`        | "Uniform"        |
| Type      | `-a ValueType=I32`      | `int32_t`        |

## Explicit List

| Axis Type | Example                         | Example Result                 |
|-----------|---------------------------------|--------------------------------|
| Int64     | `-a InputSize=[1,2,3,4,5]`      | 1, 2, 3, 4, 5                  |
| Int64Pow2 | `-a InputSize[pow2]=[4,6,8,10]` | 16, 64, 256, 1024              |
| Float64   | `-a Quality=[0.5,0.75,1.0]`     | 0.5, 0.75, 1.0                 |
| String    | `-a RNG=[Uniform,Gaussian]`     | "Uniform", "Gaussian"          |
| Type      | `-a ValueType=[U8,I32,F64]`     | `uint8_t`, `int32_t`, `double` |

## Strided Range

| Axis Type | Example                         | Example Result               |
|-----------|---------------------------------|------------------------------|
| Int64     | `-a InputSize=[2:10:2]`         | 2, 4, 6, 8, 10               |
| Int64Pow2 | `-a InputSize[pow2]=[2:10:2]`   | 4, 16, 64, 128, 256, 1024    |
| Float64   | `-a Quality=[.5:1:.1]`          | 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 |
| String    | [Not supported]                 |                              |
| Type      | [Not supported]                 |                              |
)string_bounds";

} // namespace nvbench::internal
