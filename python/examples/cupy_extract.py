import sys

import cuda.nvbench as nvbench
import cupy as cp


def as_cp_ExternalStream(
    cs: nvbench.CudaStream, dev_id: int | None = -1
) -> cp.cuda.ExternalStream:
    h = cs.addressof()
    return cp.cuda.ExternalStream(h, dev_id)


def cupy_extract_by_mask(state: nvbench.State):
    n_cols = state.get_int64("numCols")
    n_rows = state.get_int64("numRows")

    dev_id = state.get_device()
    cp_s = as_cp_ExternalStream(state.get_stream(), dev_id)

    state.collect_cupti_metrics()
    state.add_element_count(n_rows * n_cols, "# Elements")
    state.add_global_memory_reads(
        n_rows * n_cols * (cp.dtype(cp.int32).itemsize + cp.dtype("?").itemsize)
    )
    state.add_global_memory_writes(n_rows * n_cols * (cp.dtype(cp.int32).itemsize))

    with cp_s:
        X = cp.full((n_cols, n_rows), fill_value=3, dtype=cp.int32)
        mask = cp.ones((n_cols, n_rows), dtype="?")
        _ = X[mask]

    def launcher(launch: nvbench.Launch):
        with as_cp_ExternalStream(launch.get_stream(), dev_id):
            _ = X[mask]

    state.exec(launcher, sync=True)


if __name__ == "__main__":
    b = nvbench.register(cupy_extract_by_mask)
    b.add_int64_axis("numCols", [1024, 2048, 4096, 2 * 4096])
    b.add_int64_axis("numRows", [1024, 2048, 4096, 2 * 4096])

    nvbench.run_all_benchmarks(sys.argv)
