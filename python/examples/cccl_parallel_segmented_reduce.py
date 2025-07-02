import sys

import cuda.cccl.parallel.experimental.algorithms as algorithms
import cuda.cccl.parallel.experimental.iterators as iterators
import cuda.core.experimental as core
import cuda.nvbench as nvbench
import cupy as cp
import numpy as np


def as_core_Stream(cs: nvbench.CudaStream) -> core.Stream:
    return core.Stream.from_handle(cs.addressof())


def segmented_reduce(state: nvbench.State):
    "Benchmark segmented_reduce example"
    n_elems = state.getInt64("numElems")
    n_cols = state.getInt64("numCols")
    n_rows = n_elems // n_cols

    state.add_summary("numRows", n_rows)
    state.collectCUPTIMetrics()

    rng = cp.random.default_rng()
    mat = rng.integers(low=-31, high=32, dtype=np.int32, size=(n_rows, n_cols))

    def add_op(a, b):
        return a + b

    def make_scaler(step):
        def scale(row_id):
            return row_id * step

        return scale

    zero = np.int32(0)
    row_offset = make_scaler(np.int32(n_cols))
    start_offsets = iterators.TransformIterator(
        iterators.CountingIterator(zero), row_offset
    )

    end_offsets = start_offsets + 1

    d_input = mat
    h_init = np.zeros(tuple(), dtype=np.int32)
    d_output = cp.empty(n_rows, dtype=d_input.dtype)

    alg = algorithms.segmented_reduce(
        d_input, d_output, start_offsets, end_offsets, add_op, h_init
    )

    # query size of temporary storage and allocate
    temp_nbytes = alg(
        None, d_input, d_output, n_rows, start_offsets, end_offsets, h_init
    )
    temp_storage = cp.empty(temp_nbytes, dtype=cp.uint8)

    def launcher(launch: nvbench.Launch):
        s = as_core_Stream(launch.getStream())
        alg(
            temp_storage,
            d_input,
            d_output,
            n_rows,
            start_offsets,
            end_offsets,
            h_init,
            s,
        )

    state.exec(launcher)


if __name__ == "__main__":
    b = nvbench.register(segmented_reduce)
    b.addInt64Axis("numElems", [2**20, 2**22, 2**24])
    b.addInt64Axis("numCols", [1024, 2048, 4096, 8192])

    nvbench.run_all_benchmarks(sys.argv)
