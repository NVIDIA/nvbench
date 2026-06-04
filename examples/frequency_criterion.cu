/*
 *  Copyright 2026 NVIDIA Corporation
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

#include <nvbench/nvbench.cuh>

// Grab some testing kernels from NVBench:
#include <nvbench/test_kernels.cuh>

// Thrust vectors simplify memory management:
#include <thrust/device_vector.h>

#include <stdexcept>

// This example shows how to write a stopping criterion that *requires* the GPU
// clock frequency that NVBench observes for every cold-measurement sample.
//
// In addition to `do_add_measurement()`, a criterion may override
// `do_add_frequency()` to receive the SM clock rate (in Hz) measured during the
// sample. NVBench calls `add_frequency()` immediately before `add_measurement()`
// for the same sample -- but only when it can measure the clock. It is NOT
// called while profiling (the `--profile` option) or for CPU-only benchmarks
// (`nvbench::exec_tag::cpu_only` / `no_gpu`).
//
// Like the `fixed` criterion in `custom_criterion.cu`, this one simply runs for
// a fixed number of samples. The difference is that it also collects the
// per-sample frequency and throws if a sample arrives without one. The thrown
// exception is caught per-benchmark by NVBench and reported as a failure, so
// running this benchmark with `--profile` produces a clear error instead of
// silently ignoring the missing frequency.

// Inherit from the stopping_criterion_base class:
class frequency_criterion final : public nvbench::stopping_criterion_base
{
  nvbench::int64_t m_num_samples{};
  bool m_has_frequency{false};

public:
  frequency_criterion()
      : nvbench::stopping_criterion_base{"frequency", {{"max-samples", nvbench::int64_t{42}}}}
  {}

protected:
  // Setup the criterion in the `do_initialize()` method:
  virtual void do_initialize() override
  {
    m_num_samples   = 0;
    m_has_frequency = false;
  }

  // Collect the GPU clock frequency for the current sample. NVBench calls this
  // before `do_add_measurement()` whenever a frequency is available:
  virtual void do_add_frequency(nvbench::float32_t /* frequency_hz */) override
  {
    m_has_frequency = true;
  }

  // Process new measurements in the `do_add_measurement()` method:
  virtual void do_add_measurement(nvbench::float64_t /* measurement */) override
  {
    // This criterion requires a frequency for every sample. NVBench calls
    // `do_add_frequency()` before `do_add_measurement()` when one is available,
    // so a missing frequency here means none was provided for this sample:
    if (!m_has_frequency)
    {
      throw std::runtime_error(
        "frequency_criterion requires a GPU clock frequency for every sample, but none was "
        "provided. NVBench does not measure the clock frequency when profiling (--profile) or for "
        "CPU-only benchmarks (nvbench::exec_tag::cpu_only / no_gpu).");
    }

    m_has_frequency = false; // consume it; the next sample must provide its own
    m_num_samples++;
  }

  // Check if the stopping criterion is met in the `do_is_finished()` method:
  virtual bool do_is_finished() override
  {
    return m_num_samples >= m_params.get_int64("max-samples");
  }
};

// Register the criterion with NVBench:
NVBENCH_REGISTER_CRITERION(frequency_criterion);

void throughput_bench(nvbench::state &state)
{
  // Allocate input data:
  const std::size_t num_values = 64 * 1024 * 1024 / sizeof(nvbench::int32_t);
  thrust::device_vector<nvbench::int32_t> input(num_values);
  thrust::device_vector<nvbench::int32_t> output(num_values);

  // Provide throughput information:
  state.add_element_count(num_values, "NumElements");
  state.add_global_memory_reads<nvbench::int32_t>(num_values, "DataSize");
  state.add_global_memory_writes<nvbench::int32_t>(num_values);

  state.exec([&input, &output, num_values](nvbench::launch &launch) {
    (void)num_values; // clang thinks this is unused...
    nvbench::copy_kernel<<<256, 256, 0, launch.get_stream()>>>(
      thrust::raw_pointer_cast(input.data()),
      thrust::raw_pointer_cast(output.data()),
      num_values);
  });
}
NVBENCH_BENCH(throughput_bench).set_stopping_criterion("frequency");
