# Copyright 2026 NVIDIA Corporation
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

import cuda.bench as bench
import cuda.cccl.headers as headers
import cuda.core as core


def as_core_Stream(cs: bench.CudaStream) -> core.Stream:
    return core.Stream.from_handle(cs.addressof())


def make_fill_kernel():
    src = r"""
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

template <typename T>
__global__ void fill_kernel(T *buf, T value, ::cuda::std::size_t n)
{
  const auto init = blockIdx.x * blockDim.x + threadIdx.x;
  const auto step = blockDim.x * gridDim.x;

  for (auto i = init; i < n; i += step)
  {
    buf[i] = value;
  }
}
"""
    incl = headers.get_include_paths()
    opts = core.ProgramOptions(include_path=str(incl.libcudacxx))
    prog = core.Program(src, code_type="c++", options=opts)
    instance_name = "fill_kernel<::cuda::std::int32_t>"
    mod = prog.compile("cubin", name_expressions=(instance_name,))
    return mod.get_kernel(instance_name)


def make_copy_kernel():
    src = r"""
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

template <typename T>
__global__ void copy_kernel(const T *in, T *out, ::cuda::std::size_t n)
{
  const auto init = blockIdx.x * blockDim.x + threadIdx.x;
  const auto step = blockDim.x * gridDim.x;

  for (auto i = init; i < n; i += step)
  {
    out[i] = in[i];
  }
}
"""
    incl = headers.get_include_paths()
    opts = core.ProgramOptions(include_path=str(incl.libcudacxx))
    prog = core.Program(src, code_type="c++", options=opts)
    instance_name = "copy_kernel<::cuda::std::int32_t>"
    mod = prog.compile("cubin", name_expressions=(instance_name,))
    return mod.get_kernel(instance_name)


@bench.register()
@bench.axis.int64_power_of_two("Elements", range(20, 24))
def copy_with_manual_timer(state: bench.State) -> None:
    num_values = state.get_int64("Elements")
    nbytes = num_values * ctypes.sizeof(ctypes.c_int32(0))

    alloc_stream = as_core_Stream(state.get_stream())
    mem = core.DeviceMemoryResource(state.get_device())
    input_buf = mem.allocate(nbytes, alloc_stream)
    output_buf = mem.allocate(nbytes, alloc_stream)

    state.add_element_count(num_values)
    state.add_global_memory_reads(nbytes)
    state.add_global_memory_writes(nbytes)

    fill_kernel = make_fill_kernel()
    copy_kernel = make_copy_kernel()

    threads_per_block = 256
    blocks_in_grid = (num_values + threads_per_block - 1) // threads_per_block
    launch_config = core.LaunchConfig(
        grid=blocks_in_grid, block=threads_per_block, shmem_size=0
    )

    def launcher(launch: bench.Launch, timer: bench.Timer):
        stream = as_core_Stream(launch.get_stream())

        # Setup work before timer.start() is excluded from the measurement.
        core.launch(stream, launch_config, fill_kernel, input_buf, 1, num_values)
        core.launch(stream, launch_config, fill_kernel, output_buf, 0, num_values)

        # Only the copy kernel is timed.
        timer.start()
        core.launch(
            stream, launch_config, copy_kernel, input_buf, output_buf, num_values
        )
        timer.stop()

    state.exec(launcher, timer=True)


if __name__ == "__main__":
    bench.run_all_benchmarks(sys.argv)
