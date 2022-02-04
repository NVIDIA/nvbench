/*
 *  Copyright 2021-2022 NVIDIA Corporation
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

#include <nvbench/cuda_call.cuh>

#include <cuda_runtime_api.h>

namespace nvbench
{

// RAII wrapper for a cudaStream_t.
struct cuda_stream
{
  cuda_stream()
      : m_owning{true}
  {
    NVBENCH_CUDA_CALL(cudaStreamCreate(&m_stream));
  }

  cuda_stream(cudaStream_t stream, bool owning)
      : m_stream{stream}
      , m_owning{owning}
  {}

  // destroy the stream if it's owning
  void destroy()
  {
    if (m_owning)
    {
      NVBENCH_CUDA_CALL_NOEXCEPT(cudaStreamDestroy(m_stream));
    }
  }

  ~cuda_stream() { destroy(); }

  // move-only
  cuda_stream(const cuda_stream &) = delete;
  cuda_stream &operator=(const cuda_stream &) = delete;

  cuda_stream(cuda_stream &&other)
      : m_stream{other.get_stream()}
      , m_owning{other.get_owning()}
  {
    if (m_owning)
    {
      other.set_owning(not m_owning);
    }
    other.destroy();
  }

  cuda_stream &operator=(cuda_stream &&other)
  {
    m_stream = other.get_stream();
    m_owning = other.get_owning();

    if (m_owning)
    {
      other.set_owning(not m_owning);
    }
    other.destroy();

    return *this;
  }

  operator cudaStream_t() const { return m_stream; }

  cudaStream_t get_stream() const { return m_stream; }

  [[nodiscard]] bool get_owning() const { return m_owning; }
  void set_owning(bool b) { m_owning = b; }

private:
  cudaStream_t m_stream;
  bool m_owning;
};

} // namespace nvbench
