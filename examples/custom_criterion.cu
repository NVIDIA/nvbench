/*
 *  Copyright 2023 NVIDIA Corporation
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

// Inherit from the stopping_criterion class:
class fixed_criterion final : public nvbench::stopping_criterion 
{
  nvbench::int64_t m_max_samples{};
  nvbench::int64_t m_num_samples{};

public:
  // Setup the criterion in the `initialize()` method:
  virtual void initialize(const nvbench::criterion_params &params) override 
  {
    m_num_samples = 0;
    m_max_samples = params.has_value("max-samples") ? params.get_int64("max-samples") : 42;
  }

  // Process new measurements in the `add_measurement()` method:
  virtual void add_measurement(nvbench::float64_t /* measurement */) override
  {
    m_num_samples++;
  }

  // Check if the stopping criterion is met in the `is_finished()` method:
  virtual bool is_finished() override
  {
    return m_num_samples >= m_max_samples;
  }

  // Describe criterion parameters in the `get_params()` method:
  virtual const params_description &get_params() const override
  {
    static const params_description desc{
      {"max-samples", nvbench::named_values::type::int64}
    };
    return desc;
  }
};

// Register the criterion with NVBench:
static bool registered = //
  nvbench::criterion_registry::register_criterion("fixed",
                                                  std::make_unique<fixed_criterion>());

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

  state.set_stopping_criterion("fixed");

  state.exec(nvbench::exec_tag::no_batch, [&input, &output, num_values](nvbench::launch &launch) {
    nvbench::copy_kernel<<<256, 256, 0, launch.get_stream()>>>(
      thrust::raw_pointer_cast(input.data()),
      thrust::raw_pointer_cast(output.data()),
      num_values);
  });
}
NVBENCH_BENCH(throughput_bench);
