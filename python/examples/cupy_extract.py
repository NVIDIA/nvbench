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

import sys

import cuda.bench as bench
import cupy as cp


def as_cp_ExternalStream(
    cs: bench.CudaStream, dev_id: int | None = -1
) -> cp.cuda.ExternalStream:
    h = cs.addressof()
    return cp.cuda.ExternalStream(h, dev_id)


def cupy_extract_by_mask(state: bench.State):
    n_cols = state.get_int64("numCols")
    n_rows = state.get_int64("numRows")

    dev_id = state.get_device()
    cp_s = as_cp_ExternalStream(state.get_stream(), dev_id)

    state.collect_cupti_metrics()
    state.add_element_count(n_rows * n_cols, "# Elements")
    int32_dt = cp.dtype(cp.int32)
    bool_dt = cp.dtype(cp.bool_)
    state.add_global_memory_reads(
        n_rows * n_cols * (int32_dt.itemsize + bool_dt.itemsize)
    )
    state.add_global_memory_writes(n_rows * n_cols * (int32_dt.itemsize))

    with cp_s:
        X = cp.full((n_cols, n_rows), fill_value=3, dtype=int32_dt)
        mask = cp.ones((n_cols, n_rows), dtype=bool_dt)
        _ = X[mask]

    def launcher(launch: bench.Launch):
        with as_cp_ExternalStream(launch.get_stream(), dev_id):
            _ = X[mask]

    state.exec(launcher, sync=True)


if __name__ == "__main__":
    b = bench.register(cupy_extract_by_mask)
    b.add_int64_axis("numCols", [1024, 2048, 4096, 2 * 4096])
    b.add_int64_axis("numRows", [1024, 2048, 4096, 2 * 4096])

    bench.run_all_benchmarks(sys.argv)
