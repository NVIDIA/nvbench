import ctypes
import sys
from typing import Optional

import cuda.cccl.headers as headers
import cuda.core.experimental as core
import cuda.nvbench as nvbench


def as_core_Stream(cs: nvbench.CudaStream) -> core.Stream:
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


def simple(state: nvbench.State):
    state.setMinSamples(1000)
    sleep_dur = 1e-3
    krn = make_sleep_kernel()
    launch_config = core.LaunchConfig(grid=1, block=1, shmem_size=0)

    def launcher(launch: nvbench.Launch):
        s = as_core_Stream(launch.getStream())
        core.launch(s, launch_config, krn, sleep_dur)

    state.exec(launcher)


def single_float64_axis(state: nvbench.State):
    # get axis value, or default
    sleep_dur = state.getFloat64("Duration", 3.14e-4)
    krn = make_sleep_kernel()
    launch_config = core.LaunchConfig(grid=1, block=1, shmem_size=0)

    def launcher(launch: nvbench.Launch):
        s = as_core_Stream(launch.getStream())
        core.launch(s, launch_config, krn, sleep_dur)

    state.exec(launcher)


def default_value(state: nvbench.State):
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


def copy_sweep_grid_shape(state: nvbench.State):
    block_size = state.getInt64("BlockSize")
    num_blocks = state.getInt64("NumBlocks")

    # Number of int32 elements in 256MiB
    nbytes = 256 * 1024 * 1024
    num_values = nbytes // ctypes.sizeof(ctypes.c_int32(0))

    state.addElementCount(num_values)
    state.addGlobalMemoryReads(nbytes)
    state.addGlobalMemoryWrites(nbytes)

    dev_id = state.getDevice()
    alloc_s = as_core_Stream(state.getStream())
    input_buf = core.DeviceMemoryResource(dev_id).allocate(nbytes, alloc_s)
    output_buf = core.DeviceMemoryResource(dev_id).allocate(nbytes, alloc_s)

    krn = make_copy_kernel()
    launch_config = core.LaunchConfig(grid=num_blocks, block=block_size, shmem_size=0)

    def launcher(launch: nvbench.Launch):
        s = as_core_Stream(launch.getStream())
        core.launch(s, launch_config, krn, input_buf, output_buf, num_values)

    state.exec(launcher)


def copy_type_sweep(state: nvbench.State):
    type_id = state.getInt64("TypeID")

    types_map = {
        0: (ctypes.c_uint8, "::cuda::std::uint8_t"),
        1: (ctypes.c_uint16, "::cuda::std::uint16_t"),
        2: (ctypes.c_uint32, "::cuda::std::uint32_t"),
        3: (ctypes.c_uint64, "::cuda::std::uint64_t"),
        4: (ctypes.c_float, "float"),
        5: (ctypes.c_double, "double"),
    }

    value_ctype, value_cuda_t = types_map[type_id]
    state.add_summary("Type", value_cuda_t)

    # Number of elements in 256MiB
    nbytes = 256 * 1024 * 1024
    num_values = nbytes // ctypes.sizeof(value_ctype(0))

    state.addElementCount(num_values)
    state.addGlobalMemoryReads(nbytes)
    state.addGlobalMemoryWrites(nbytes)

    dev_id = state.getDevice()
    alloc_s = as_core_Stream(state.getStream())
    input_buf = core.DeviceMemoryResource(dev_id).allocate(nbytes, alloc_s)
    output_buf = core.DeviceMemoryResource(dev_id).allocate(nbytes, alloc_s)

    krn = make_copy_kernel(value_cuda_t, value_cuda_t)
    launch_config = core.LaunchConfig(grid=256, block=256, shmem_size=0)

    def launcher(launch: nvbench.Launch):
        s = as_core_Stream(launch.getStream())
        core.launch(s, launch_config, krn, input_buf, output_buf, num_values)

    state.exec(launcher)


if __name__ == "__main__":
    # Benchmark without axes
    nvbench.register(simple)

    # benchmark with no axes, that uses default value
    nvbench.register(default_value)
    # specify axis
    nvbench.register(single_float64_axis).addFloat64Axis("Duration", [7e-5, 1e-4, 5e-4])

    copy1_bench = nvbench.register(copy_sweep_grid_shape)
    copy1_bench.addInt64Axis("BlockSize", [2**x for x in range(6, 10, 2)])
    copy1_bench.addInt64Axis("NumBlocks", [2**x for x in range(6, 10, 2)])

    copy2_bench = nvbench.register(copy_type_sweep)
    copy2_bench.addInt64Axis("TypeID", range(0, 6))

    nvbench.run_all_benchmarks(sys.argv)
