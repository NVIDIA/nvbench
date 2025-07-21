import sys

import cuda.cccl.headers as headers
import cuda.core.experimental as core
import cuda.nvbench as nvbench


def as_core_Stream(cs: nvbench.CudaStream) -> core.Stream:
    "Create view into native stream provided by NVBench"
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


def runtime_skip(state: nvbench.State):
    duration = state.get_float64("Duration")
    kramble = state.get_string("Kramble")

    # Skip Baz benchmarks with 0.8 ms duration
    if kramble == "Baz" and duration < 0.8e-3:
        state.skip("Short 'Baz' benchmarks are skipped")
        return

    # Skip Foo benchmark with > 0.3 ms duration
    if kramble == "Foo" and duration > 0.3e-3:
        state.skip("Long 'Foo' benchmarks are skipped")
        return

    krn = make_sleep_kernel()
    launch_cfg = core.LaunchConfig(grid=1, block=1, shmem_size=0)

    def launcher(launch: nvbench.Launch):
        s = as_core_Stream(launch.get_stream())
        core.launch(s, launch_cfg, krn, duration)

    state.exec(launcher)


if __name__ == "__main__":
    b = nvbench.register(runtime_skip)
    b.add_float64_axis("Duration", [1e-4 + k * 0.25e-3 for k in range(5)])
    b.add_string_axis("Kramble", ["Foo", "Bar", "Baz"])

    nvbench.run_all_benchmarks(sys.argv)
