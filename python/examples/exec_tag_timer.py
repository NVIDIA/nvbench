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


def make_copy_kernel():
    src = r"""
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

template <typename T>
__global__ void copy_kernel(const T *in, T *out, cuda::std::size_t n)
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
    instance_name = "copy_kernel<cuda::std::int32_t>"
    mod = prog.compile("cubin", name_expressions=(instance_name,))
    return mod.get_kernel(instance_name)


def make_sequence_kernel():
    src = r"""
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

template <typename T>
__global__ void sequence_kernel(T *buf, cuda::std::size_t n)
{
  const auto init = blockIdx.x * blockDim.x + threadIdx.x;
  const auto step = blockDim.x * gridDim.x;

  for (auto i = init; i < n; i += step)
  {
    buf[i] = static_cast<T>(i);
  }
}
"""
    incl = headers.get_include_paths()
    opts = core.ProgramOptions(include_path=str(incl.libcudacxx))
    prog = core.Program(src, code_type="c++", options=opts)
    instance_name = "sequence_kernel<cuda::std::int32_t>"
    mod = prog.compile("cubin", name_expressions=(instance_name,))
    return mod.get_kernel(instance_name)


def make_mod2_inplace_kernel():
    src = r"""
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

__global__ void mod2_inplace_kernel(cuda::std::int32_t *data,
                                    cuda::std::size_t n)
{
  const auto init = blockIdx.x * blockDim.x + threadIdx.x;
  const auto step = blockDim.x * gridDim.x;

  for (auto i = init; i < n; i += step)
  {
    data[i] = data[i] % 2;
  }
}
"""
    incl = headers.get_include_paths()
    opts = core.ProgramOptions(include_path=str(incl.libcudacxx))
    prog = core.Program(src, code_type="c++", options=opts)
    mod = prog.compile("cubin", name_expressions=("mod2_inplace_kernel",))
    return mod.get_kernel("mod2_inplace_kernel")


@bench.register()
def mod2_inplace(state: bench.State) -> None:
    num_values = 64 * 1024 * 1024 // ctypes.sizeof(ctypes.c_int32(0))
    nbytes = num_values * ctypes.sizeof(ctypes.c_int32(0))

    alloc_stream = as_core_Stream(state.get_stream())
    mem = core.DeviceMemoryResource(state.get_device())
    input_buf = mem.allocate(nbytes, alloc_stream)
    data_buf = mem.allocate(nbytes, alloc_stream)

    state.add_element_count(num_values)
    state.add_global_memory_reads(nbytes)
    state.add_global_memory_writes(nbytes)

    sequence_kernel = make_sequence_kernel()
    copy_kernel = make_copy_kernel()
    mod2_kernel = make_mod2_inplace_kernel()

    threads_per_block = 256
    blocks_in_grid = (num_values + threads_per_block - 1) // threads_per_block
    launch_config = core.LaunchConfig(
        grid=blocks_in_grid, block=threads_per_block, shmem_size=0
    )

    core.launch(alloc_stream, launch_config, sequence_kernel, input_buf, num_values)

    def launcher(launch: bench.Launch, timer: bench.Timer):
        stream = as_core_Stream(launch.get_stream())

        # Reset working data before timing the in-place operation.
        core.launch(stream, launch_config, copy_kernel, input_buf, data_buf, num_values)

        timer.start()
        core.launch(stream, launch_config, mod2_kernel, data_buf, num_values)
        timer.stop()

    state.exec(launcher, timer=True)


if __name__ == "__main__":
    bench.run_all_benchmarks(sys.argv)
