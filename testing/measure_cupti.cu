// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <nvbench/benchmark.cuh>
#include <nvbench/cuda_call.cuh>
#include <nvbench/detail/measure_cupti.cuh>
#include <nvbench/detail/state_generator.cuh>
#include <nvbench/device_info.cuh>
#include <nvbench/device_manager.cuh>
#include <nvbench/state.cuh>

#include <cuda_runtime_api.h>

#include <fmt/format.h>

#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

#include "test_asserts.cuh"

namespace
{

struct dummy_generator
{
  template <typename TypeConfig>
  void operator()(nvbench::state &, TypeConfig)
  {}
};

__global__ void empty_kernel() {}

std::size_t get_persisting_l2_limit()
{
  std::size_t limit{};
  NVBENCH_CUDA_CALL(cudaDeviceGetLimit(&limit, cudaLimitPersistingL2CacheSize));
  return limit;
}

struct restore_limit
{
  std::size_t original{get_persisting_l2_limit()};

  ~restore_limit()
  {
    NVBENCH_CUDA_CALL_NOEXCEPT(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, original));
  }
};

struct launcher_error : std::runtime_error
{
  launcher_error()
      : std::runtime_error{"CUPTI launcher test exception"}
  {}
};

void test_persisting_l2_cache(const nvbench::device_info &device,
                              std::size_t requested_limit,
                              bool disable_cache,
                              bool throw_from_launcher)
{
  NVBENCH_CUDA_CALL(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, requested_limit));
  const auto original_limit = get_persisting_l2_limit();
  ASSERT(original_limit > 0);

  nvbench::benchmark<dummy_generator> benchmark;
  benchmark.set_devices(std::vector<nvbench::device_info>{device});
  auto states = nvbench::detail::state_generator::create(benchmark);
  ASSERT(states.size() == 1);
  auto &state = states.front();
  state.collect_l2_hit_rates();
  state.set_disable_persisting_l2_cache(disable_cache);

  int launches{};
  auto launcher = [&](nvbench::launch &launch, auto &timer) {
    // Re-seed before every replay to ensure each timer start applies the option.
    NVBENCH_CUDA_CALL(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, original_limit));
    timer.start();
    const auto measured_limit = get_persisting_l2_limit();
    empty_kernel<<<1, 1, 0, launch.get_stream()>>>();
    NVBENCH_CUDA_CALL(cudaGetLastError());
    timer.stop();
    ++launches;
    ASSERT(measured_limit == (disable_cache ? std::size_t{0} : original_limit));
    if (throw_from_launcher)
    {
      throw launcher_error{};
    }
  };

  // Keep the measurement object alive while checking restoration: cleanup must
  // happen when the replay phase exits, not only when its profiler is destroyed.
  nvbench::detail::measure_cupti<decltype(launcher)> measure{state, launcher};
  bool caught{};
  try
  {
    measure();
  }
  catch (const launcher_error &)
  {
    caught = true;
  }
  ASSERT(launches > 0);
  ASSERT(caught == throw_from_launcher);
  ASSERT(get_persisting_l2_limit() == original_limit);
}

} // namespace

int main()
try
{
  std::size_t tested_devices{};
  for (const auto &device : nvbench::device_manager::get().get_devices())
  {
    device.set_active();
    int maximum{};
    NVBENCH_CUDA_CALL(
      cudaDeviceGetAttribute(&maximum, cudaDevAttrMaxPersistingL2CacheSize, device.get_id()));
    if (maximum == 0 || !device.is_cupti_supported())
    {
      fmt::print("Skipping device {}: requires both persisting L2 and CUPTI support.\n",
                 device.get_id());
      continue;
    }

    try
    {
      const restore_limit restore;
      const auto requested_limit = static_cast<std::size_t>(maximum / 2);
      for (bool disable_cache : {false, true})
      {
        for (bool throw_from_launcher : {false, true})
        {
          test_persisting_l2_cache(device, requested_limit, disable_cache, throw_from_launcher);
        }
      }
      ++tested_devices;
    }
    catch (const std::exception &error)
    {
      if (std::string{error.what()}.find("CUPTI_ERROR_INSUFFICIENT_PRIVILEGES") ==
          std::string::npos)
      {
        throw;
      }
      fmt::print("Skipping device {}: GPU performance counters require profiling privileges.\n",
                 device.get_id());
    }
  }

  if (tested_devices == 0)
  {
    fmt::print("Skipping: no available device could run the CUPTI cache tests.\n");
    return 77;
  }
  return 0;
}
catch (const std::exception &error)
{
  fmt::print("{}\n", error.what());
  return 1;
}
