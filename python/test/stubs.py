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
from typing import Dict, Optional, Tuple

import cuda.bench as bench
import cuda.cccl.headers as headers
import cuda.core.experimental as core


def as_core_Stream(cs: bench.CudaStream) -> core.Stream:
    return core.Stream.from_handle(cs.addressof())


def make_sleep_kernel():
    """JITs sleep_kernel(seconds)"""
    src = r"""
#include <cuda/std/cstdint>
#include <cuda/std/chrono>

// Each launched thread just sleeps for `seconds`.
__global__ void sleep_kernel(double seconds) {
  namespace chrono = ::cuda::std::chrono;
  using hr_clock = chrono::high_resolution_clock;

  auto duration = static_cast<cuda::std::int64_t>(seconds * 1e9);
  const auto ns = chrono::nanoseconds(duration);

  const auto start = hr_clock::now();
  const auto finish = start + ns;

  auto now = hr_clock::now();
  while (now < finish)
  {
    now = hr_clock::now();
  }
}
"""
    incl = headers.get_include_paths()
    opts = core.ProgramOptions(include_path=str(incl.libcudacxx))
    prog = core.Program(src, code_type="c++", options=opts)
    mod = prog.compile("cubin", name_expressions=("sleep_kernel",))
    return mod.get_kernel("sleep_kernel")


def no_axes(state: bench.State):
    state.set_min_samples(1000)
    sleep_dur = 1e-3
    krn = make_sleep_kernel()
    launch_config = core.LaunchConfig(grid=1, block=1, shmem_size=0)

    print(f"Stopping criterion used: {state.get_stopping_criterion()}")

    def launcher(launch: bench.Launch):
        s = as_core_Stream(launch.get_stream())
        core.launch(s, launch_config, krn, sleep_dur)

    state.exec(launcher)


def tags(state: bench.State):
    state.set_min_samples(1000)
    sleep_dur = 1e-3
    krn = make_sleep_kernel()
    launch_config = core.LaunchConfig(grid=1, block=1, shmem_size=0)

    sync_flag = bool(state.get_int64("Sync"))
    batched_flag = bool(state.get_int64("Batched"))

    def launcher(launch: bench.Launch):
        s = as_core_Stream(launch.get_stream())
        core.launch(s, launch_config, krn, sleep_dur)

    state.exec(launcher, sync=sync_flag, batched=batched_flag)


def single_float64_axis(state: bench.State):
    # get axis value, or default
    default_sleep_dur = 3.14e-4
    sleep_dur = state.get_float64_or_default("Duration", default_sleep_dur)
    krn = make_sleep_kernel()
    launch_config = core.LaunchConfig(grid=1, block=1, shmem_size=0)

    def launcher(launch: bench.Launch):
        s = as_core_Stream(launch.get_stream())
        core.launch(s, launch_config, krn, sleep_dur)

    state.exec(launcher)


def default_value(state: bench.State):
    single_float64_axis(state)


def make_copy_kernel(in_type: Optional[str] = None, out_type: Optional[str] = None):
    src = r"""
#include <cuda/std/cstdint>
#include <cuda/std/cstddef>
/*!
 * Naive copy of `n` values from `in` -> `out`.
 */
template <typename T, typename U>
__global__ void copy_kernel(const T *in, U *out, ::cuda::std::size_t n)
{
  const auto init = blockIdx.x * blockDim.x + threadIdx.x;
  const auto step = blockDim.x * gridDim.x;

  for (auto i = init; i < n; i += step)
  {
    out[i] = static_cast<U>(in[i]);
  }
}
"""
    incl = headers.get_include_paths()
    opts = core.ProgramOptions(include_path=str(incl.libcudacxx))
    prog = core.Program(src, code_type="c++", options=opts)
    if in_type is None:
        in_type = "::cuda::std::int32_t"
    if out_type is None:
        out_type = "::cuda::std::int32_t"
    instance_name = f"copy_kernel<{in_type}, {out_type}>"
    mod = prog.compile("cubin", name_expressions=(instance_name,))
    return mod.get_kernel(instance_name)


def copy_sweep_grid_shape(state: bench.State):
    block_size = state.get_int64("BlockSize")
    num_blocks = state.get_int64("NumBlocks")

    # Number of int32 elements in 256MiB
    nbytes = 256 * 1024 * 1024
    num_values = nbytes // ctypes.sizeof(ctypes.c_int32(0))

    state.add_element_count(num_values)
    state.add_global_memory_reads(nbytes)
    state.add_global_memory_writes(nbytes)

    dev_id = state.get_device()
    alloc_s = as_core_Stream(state.get_stream())
    input_buf = core.DeviceMemoryResource(dev_id).allocate(nbytes, alloc_s)
    output_buf = core.DeviceMemoryResource(dev_id).allocate(nbytes, alloc_s)

    krn = make_copy_kernel()
    launch_config = core.LaunchConfig(grid=num_blocks, block=block_size, shmem_size=0)

    def launcher(launch: bench.Launch):
        s = as_core_Stream(launch.get_stream())
        core.launch(s, launch_config, krn, input_buf, output_buf, num_values)

    state.exec(launcher)


def copy_type_sweep(state: bench.State):
    type_id = state.get_int64("TypeID")

    types_map: Dict[int, Tuple[type, str]] = {
        0: (ctypes.c_uint8, "cuda::std::uint8_t"),
        1: (ctypes.c_uint16, "cuda::std::uint16_t"),
        2: (ctypes.c_uint32, "cuda::std::uint32_t"),
        3: (ctypes.c_uint64, "cuda::std::uint64_t"),
        4: (ctypes.c_float, "float"),
        5: (ctypes.c_double, "double"),
    }

    value_ctype, value_cuda_t = types_map[type_id]
    state.add_summary("Type", value_cuda_t)

    # Number of elements in 256MiB
    nbytes = 256 * 1024 * 1024
    num_values = nbytes // ctypes.sizeof(value_ctype)

    state.add_element_count(num_values)
    state.add_global_memory_reads(nbytes)
    state.add_global_memory_writes(nbytes)

    dev_id = state.get_device()
    alloc_s = as_core_Stream(state.get_stream())
    input_buf = core.DeviceMemoryResource(dev_id).allocate(nbytes, alloc_s)
    output_buf = core.DeviceMemoryResource(dev_id).allocate(nbytes, alloc_s)

    krn = make_copy_kernel(value_cuda_t, value_cuda_t)
    launch_config = core.LaunchConfig(grid=256, block=256, shmem_size=0)

    def launcher(launch: bench.Launch):
        s = as_core_Stream(launch.get_stream())
        core.launch(s, launch_config, krn, input_buf, output_buf, num_values)

    state.exec(launcher)


if __name__ == "__main__":
    # Benchmark without axes
    simple_b = bench.register(no_axes)
    simple_b.set_stopping_criterion("entropy")
    simple_b.set_criterion_param_int64("unused_int", 100)

    tags_b = bench.register(tags)
    tags_b.add_int64_axis("Sync", [0, 1])
    tags_b.add_int64_axis("Batched", [0, 1])

    # benchmark with no axes, that uses default value
    default_b = bench.register(default_value)
    default_b.set_min_samples(7)

    # specify axis
    axes_b = bench.register(single_float64_axis).add_float64_axis(
        "Duration", [7e-5, 1e-4, 5e-4]
    )
    axes_b.set_timeout(20)
    axes_b.set_skip_time(1e-5)
    axes_b.set_throttle_threshold(0.2)
    axes_b.set_throttle_recovery_delay(0.1)

    copy1_bench = bench.register(copy_sweep_grid_shape)
    copy1_bench.add_int64_power_of_two_axis("BlockSize", range(6, 10, 2))
    copy1_bench.add_int64_axis("NumBlocks", [2**x for x in range(6, 10, 2)])

    copy2_bench = bench.register(copy_type_sweep)
    copy2_bench.add_int64_axis("TypeID", range(0, 6))

    bench.run_all_benchmarks(sys.argv)
