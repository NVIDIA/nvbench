import sys

import cuda.nvbench as nvbench
import numpy as np
from numba import cuda


@cuda.jit()
def kernel(a, b, c):
    tid = cuda.grid(1)
    size = len(a)

    if tid < size:
        c[tid] = a[tid] + b[tid]


def getNumbaStream(launch):
    return cuda.external_stream(launch.getStream().addressof())


def add_two(state):
    # state.skip("Skipping this benchmark for no reason")
    N = state.getInt64("elements")
    a = cuda.to_device(np.random.random(N))
    c = cuda.device_array_like(a)

    state.addGlobalMemoryReads(a.nbytes)
    state.addGlobalMemoryWrites(c.nbytes)

    nthreads = 256
    nblocks = (len(a) + nthreads - 1) // nthreads

    # First call locks, can't use async benchmarks until sync tag is supported
    kernel[nblocks, nthreads](a, a, c)
    cuda.synchronize()

    def kernel_launcher(launch):
        stream = getNumbaStream(launch)
        kernel[nblocks, nthreads, stream](a, a, c)

    state.exec(kernel_launcher, batched=True, sync=True)


def add_float(state):
    N = state.getInt64("elements")
    v = state.getFloat64("v")
    name = state.getString("name")
    a = cuda.to_device(np.random.random(N).astype(np.float32))
    b = cuda.to_device(np.random.random(N).astype(np.float32))
    c = cuda.device_array_like(a)

    state.addGlobalMemoryReads(a.nbytes + b.nbytes)
    state.addGlobalMemoryWrites(c.nbytes)

    nthreads = 64
    nblocks = (len(a) + nthreads - 1) // nthreads

    def kernel_launcher(launch):
        _ = v
        _ = name
        stream = getNumbaStream(launch)
        kernel[nblocks, nthreads, stream](a, b, c)

    state.exec(kernel_launcher, batched=True, sync=True)


def add_three(state):
    N = state.getInt64("elements")
    a = cuda.to_device(np.random.random(N).astype(np.float32))
    b = cuda.to_device(np.random.random(N).astype(np.float32))
    c = cuda.device_array_like(a)

    state.addGlobalMemoryReads(a.nbytes + b.nbytes)
    state.addGlobalMemoryWrites(c.nbytes)

    nthreads = 256
    nblocks = (len(a) + nthreads - 1) // nthreads

    def kernel_launcher(launch):
        stream = getNumbaStream(launch)
        kernel[nblocks, nthreads, stream](a, b, c)

    state.exec(kernel_launcher, batched=True, sync=True)
    cuda.synchronize()


def register_benchmarks():
    (
        nvbench.register(add_two).addInt64Axis(
            "elements", [2**pow2 for pow2 in range(20, 23)]
        )
    )
    (
        nvbench.register(add_float)
        .addFloat64Axis("v", [0.1, 0.3])
        .addStringAxis("name", ["Anne", "Lynda"])
        .addInt64Axis("elements", [2**pow2 for pow2 in range(20, 23)])
    )
    (
        nvbench.register(add_three).addInt64Axis(
            "elements", [2**pow2 for pow2 in range(20, 22)]
        )
    )


if __name__ == "__main__":
    register_benchmarks()
    nvbench.run_all_benchmarks(sys.argv)
