import sys
import time

import cuda.cccl.headers as headers
import cuda.core.experimental as core
import cuda.nvbench as nvbench

host_sleep_duration = 0.1


def cpu_only_sleep_bench(state: nvbench.State) -> None:
    def launcher(launch: nvbench.Launch):
        time.sleep(host_sleep_duration)

    state.exec(launcher)


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


def mixed_sleep_bench(state: nvbench.State) -> None:
    sync = state.get_string("Sync")
    sync_flag = sync == "Do sync"

    gpu_sleep_dur = 225e-3
    krn = make_sleep_kernel()
    launch_config = core.LaunchConfig(grid=1, block=1, shmem_size=0)

    def launcher(launch: nvbench.Launch):
        # host overhead
        time.sleep(host_sleep_duration)
        # GPU computation
        s = as_core_Stream(launch.get_stream())
        core.launch(s, launch_config, krn, gpu_sleep_dur)

    state.exec(launcher, sync=sync_flag)


if __name__ == "__main__":
    # time function only doing work (sleeping) on the host
    # using CPU timer only
    b = nvbench.register(cpu_only_sleep_bench)
    b.set_is_cpu_only(True)

    # time the function that does work on both GPU and CPU
    b2 = nvbench.register(mixed_sleep_bench)
    b2.add_string_axis("Sync", ["Do not sync", "Do sync"])

    nvbench.run_all_benchmarks(sys.argv)
