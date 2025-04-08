/*
 *  Copyright 2025 NVIDIA Corporation
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

#include <nvbench/types.cuh>

namespace nvbench
{

struct cuda_stream;

namespace detail
{

struct timestamps_kernel
{
  timestamps_kernel();
  ~timestamps_kernel();

  void record(const nvbench::cuda_stream &stream);

  // move-only
  timestamps_kernel(const timestamps_kernel &)            = delete;
  timestamps_kernel(timestamps_kernel &&)                 = default;
  timestamps_kernel &operator=(const timestamps_kernel &) = delete;
  timestamps_kernel &operator=(timestamps_kernel &&)      = default;

  nvbench::uint64_t m_host_timestamps[2];
  nvbench::uint64_t *m_device_timestamps{};
};

} // namespace detail

} // namespace nvbench
