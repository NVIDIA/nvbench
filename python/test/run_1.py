import sys

import cuda.bench as bench
import numpy as np
from numba import cuda


@cuda.jit()
def kernel(a, b, c):
    tid = cuda.grid(1)
    size = len(a)

    if tid < size:
        c[tid] = a[tid] + b[tid]


def get_numba_stream(launch: bench.Launch):
    return cuda.external_stream(launch.get_stream().addressof())


def skipit(state: bench.State) -> None:
    state.skip("Skipping this benchmark for no reason")


def add_two(state: bench.State):
    N = state.get_int64("elements")
    a = cuda.to_device(np.random.random(N))
    c = cuda.device_array_like(a)

    assert "elements" in state.get_axis_values()
    assert "elements=" in state.get_axis_values_as_string()

    state.add_global_memory_reads(a.nbytes)
    state.add_global_memory_writes(c.nbytes)

    nthreads = 256
    nblocks = (len(a) + nthreads - 1) // nthreads

    # First call locks, can't use async benchmarks until sync tag is supported
    kernel[nblocks, nthreads](a, a, c)
    cuda.synchronize()

    def kernel_launcher(launch):
        stream = get_numba_stream(launch)
        kernel[nblocks, nthreads, stream](a, a, c)

    state.exec(kernel_launcher, batched=True, sync=True)


def add_float(state: bench.State):
    N = state.get_int64("elements")
    v = state.get_float64("v")
    name = state.get_string("name")
    a = cuda.to_device(np.random.random(N).astype(np.float32))
    b = cuda.to_device(np.random.random(N).astype(np.float32))
    c = cuda.device_array_like(a)

    state.add_global_memory_reads(a.nbytes + b.nbytes)
    state.add_global_memory_writes(c.nbytes)

    nthreads = 64
    nblocks = (len(a) + nthreads - 1) // nthreads

    axis_values = state.get_axis_values()
    assert "elements" in axis_values
    assert "v" in axis_values
    assert "name" in axis_values
    assert axis_values["elements"] == N
    assert axis_values["v"] == v
    assert axis_values["name"] == name

    def kernel_launcher(launch):
        _ = v
        _ = name
        stream = get_numba_stream(launch)
        kernel[nblocks, nthreads, stream](a, b, c)

    state.exec(kernel_launcher, batched=True, sync=True)


def add_three(state: bench.State):
    N = state.get_int64("elements")
    a = cuda.to_device(np.random.random(N).astype(np.float32))
    b = cuda.to_device(np.random.random(N).astype(np.float32))
    c = cuda.device_array_like(a)

    state.add_global_memory_reads(a.nbytes + b.nbytes)
    state.add_global_memory_writes(c.nbytes)

    nthreads = 256
    nblocks = (len(a) + nthreads - 1) // nthreads

    def kernel_launcher(launch):
        stream = get_numba_stream(launch)
        kernel[nblocks, nthreads, stream](a, b, c)

    state.exec(kernel_launcher, batched=True, sync=True)
    cuda.synchronize()


def register_benchmarks():
    (
        bench.register(add_two).add_int64_axis(
            "elements", [2**pow2 - 1 for pow2 in range(20, 23)]
        )
    )
    (
        bench.register(add_float)
        .add_float64_axis("v", [0.1, 0.3])
        .add_string_axis("name", ["Anne", "Lynda"])
        .add_int64_power_of_two_axis("elements", range(20, 23))
    )
    bench.register(add_three).add_int64_power_of_two_axis("elements", range(20, 22))
    bench.register(skipit)


if __name__ == "__main__":
    register_benchmarks()
    bench.run_all_benchmarks(sys.argv)
