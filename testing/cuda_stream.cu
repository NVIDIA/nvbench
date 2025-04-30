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

#include <nvbench/config.cuh>
#include <nvbench/cuda_stream.cuh>
#include <nvbench/device_manager.cuh>
#include <nvbench/types.cuh>

#include <fmt/format.h>

#include "test_asserts.cuh"

namespace
{
#ifdef NVBENCH_HAS_CUPTI
/**
 * @brief Queries and returns the device id that the given \p cuda_stream is associated with
 *
 * @param cuda_stream The stream to get the device id for
 * @return The device id that \p cuda_stream is associated with
 */
int get_device_of_stream(cudaStream_t cuda_stream)
{
  CUcontext ctx;
  NVBENCH_DRIVER_API_CALL(cuStreamGetCtx(CUstream{cuda_stream}, &ctx));
  NVBENCH_DRIVER_API_CALL(cuCtxPushCurrent(ctx));
  CUdevice device_id{};
  NVBENCH_DRIVER_API_CALL(cuCtxGetDevice(&device_id));
  NVBENCH_DRIVER_API_CALL(cuCtxPopCurrent(&ctx));
  return static_cast<int>(device_id);
}
#endif
} // namespace

void test_basic()
{
#ifdef NVBENCH_HAS_CUPTI
  // Get devices
  auto devices = nvbench::device_manager::get().get_devices();

  // Iterate over devices
  for (auto const &device_info : devices)
  {
    // Create stream on the device before it becomes the active device
    nvbench::cuda_stream device_stream(device_info);

    // Verify cuda stream is associated with the correct cuda device
    ASSERT(get_device_of_stream(device_stream.get_stream()) == device_info.get_id());

    // Set the device as active device
    device_info.set_active();

    // Create the stream (implicitly) on the device that is currently active
    nvbench::cuda_stream current_device_stream{};

    // Verify the cuda stream was in fact associated with the currently active device
    ASSERT(get_device_of_stream(current_device_stream.get_stream()) == device_info.get_id());
  }
#endif
}

int main() { test_basic(); }
