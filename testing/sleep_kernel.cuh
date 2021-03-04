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

#pragma once

#include <cuda/std/chrono>

#include <cuda_runtime.h>

__global__ void sleep_kernel(double seconds)
{
  const auto start = cuda::std::chrono::high_resolution_clock::now();
  const auto ns    = cuda::std::chrono::nanoseconds(
    static_cast<nvbench::int64_t>(seconds * 1000 * 1000 * 1000));
  const auto finish = start + ns;

  auto now = cuda::std::chrono::high_resolution_clock::now();
  while (now < finish)
  {
    now = cuda::std::chrono::high_resolution_clock::now();
  }
}
