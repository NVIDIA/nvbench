# Copyright 2025 NVIDIA Corporation
#
#  Licensed under the Apache License, Version 2.0 with the LLVM exception
#  (the "License"); you may not use this file except in compliance with
#  the License.
#
#  You may obtain a copy of the License at
#
#      http://llvm.org/foundation/relicensing/LICENSE.txt
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import ctypes
import sys
from typing import Optional

import cuda.bench as bench
import cuda.cccl.headers as headers
import cuda.core.experimental as core


def as_core_Stream(cs: bench.CudaStream) -> core.Stream:
    "Create view of native stream used by NVBench"
    return core.Stream.from_handle(cs.addressof())


def make_fill_kernel(data_type: Optional[str] = None):
    src = r"""
#include <cuda/std/cstdint>
#include <cuda/std/cstddef>
/*!
 * Naive setting of values in buffer
 */
template <typename T>
__global__ void fill_kernel(T *buf, T v, ::cuda::std::size_t n)
{
  const auto init = blockIdx.x * blockDim.x + threadIdx.x;
  const auto step = blockDim.x * gridDim.x;

  for (auto i = init; i < n; i += step)
  {
    buf[i] = v;
  }
}
"""
    incl = headers.get_include_paths()
    opts = core.ProgramOptions(include_path=str(incl.libcudacxx))
    prog = core.Program(src, code_type="c++", options=opts)
    if data_type is None:
        data_type = "::cuda::std::int32_t"
    instance_name = f"fill_kernel<{data_type}>"
    mod = prog.compile("cubin", name_expressions=(instance_name,))
    return mod.get_kernel(instance_name)


def synchronizing_bench(state: bench.State):
    n_values = 64 * 1024 * 1024
    n_bytes = n_values * ctypes.sizeof(ctypes.c_int32(0))

    alloc_s = as_core_Stream(state.get_stream())
    buffer = core.DeviceMemoryResource(state.get_device()).allocate(n_bytes, alloc_s)

    state.add_element_count(n_values, "Items")
    state.add_global_memory_writes(n_bytes, "Size")

    krn = make_fill_kernel()
    launch_config = core.LaunchConfig(grid=256, block=256, shmem_size=0)

    def launcher(launch: bench.Launch):
        s = as_core_Stream(launch.get_stream())
        core.launch(s, launch_config, krn, buffer, 0, n_values)
        s.sync()

    # since launcher contains synchronization point,
    # setting sync=True is required to avoid a deadlock
    state.exec(launcher, sync=True)


if __name__ == "__main__":
    bench.register(synchronizing_bench)
    bench.run_all_benchmarks(sys.argv)
