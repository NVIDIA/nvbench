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
import cuda.bindings.driver as driver
import cuda.core.experimental as core
import cupy as cp
import cutlass
import numpy as np


def as_bindings_Stream(cs: bench.CudaStream) -> driver.CUstream:
    return driver.CUstream(cs.addressof())


def as_core_Stream(cs: bench.CudaStream) -> core.Stream:
    return core.Stream.from_handle(cs.addressof())


def make_cp_array(
    arr_h: np.ndarray, dev_buf: core.Buffer, dev_id: int | None
) -> cp.ndarray:
    cp_memview = cp.cuda.UnownedMemory(
        int(dev_buf.handle), dev_buf.size, dev_buf, -1 if dev_id is None else dev_id
    )
    zero_offset = 0
    return cp.ndarray(
        arr_h.shape,
        dtype=arr_h.dtype,
        memptr=cp.cuda.MemoryPointer(cp_memview, zero_offset),
    )


def cutlass_gemm(state: bench.State) -> None:
    n = state.get_int64("N")
    r = state.get_int64("R")

    alpha = state.get_float64("alpha")

    dt = np.float64
    A_h = np.random.randn(n, r).astype(dt)
    B_h = np.copy(A_h.mT)
    C_h = np.eye(n, dtype=dt)
    D_h = np.zeros_like(C_h)

    if n >= 1024:
        # allow more time for large inputs
        state.set_timeout(360)

    dev_id = state.get_device()
    cs = state.get_stream()
    s = as_bindings_Stream(cs)
    core_s = as_core_Stream(cs)

    A_d = core.DeviceMemoryResource(dev_id).allocate(A_h.nbytes, core_s)
    B_d = core.DeviceMemoryResource(dev_id).allocate(B_h.nbytes, core_s)
    C_d = core.DeviceMemoryResource(dev_id).allocate(C_h.nbytes, core_s)
    D_d = core.DeviceMemoryResource(dev_id).allocate(D_h.nbytes, core_s)

    driver.cuMemcpyAsync(A_d.handle, A_h.ctypes.data, A_h.nbytes, s)
    driver.cuMemcpyAsync(B_d.handle, B_h.ctypes.data, B_h.nbytes, s)
    driver.cuMemcpyAsync(C_d.handle, C_h.ctypes.data, C_h.nbytes, s)
    driver.cuMemcpyAsync(D_d.handle, D_h.ctypes.data, D_h.nbytes, s)

    A_cp = make_cp_array(A_h, A_d, dev_id)
    B_cp = make_cp_array(B_h, B_d, dev_id)
    C_cp = make_cp_array(C_h, C_d, dev_id)
    D_cp = make_cp_array(D_h, D_d, dev_id)

    plan = cutlass.op.Gemm(
        A=A_cp,
        B=B_cp,
        C=C_cp,
        D=D_cp,
        element=dt,
        alpha=alpha,
        beta=1,
        layout=cutlass.LayoutType.RowMajor,
    )
    # warm-up to ensure compilation is not timed
    plan.run(stream=s)

    def launcher(launch: bench.Launch) -> None:
        s = as_bindings_Stream(launch.get_stream())
        plan.run(stream=s, sync=False)

    state.exec(launcher)


if __name__ == "__main__":
    gemm_b = bench.register(cutlass_gemm)
    gemm_b.add_int64_axis("R", [16, 64, 256])
    gemm_b.add_int64_axis("N", [256, 512, 1024, 2048])

    gemm_b.add_float64_axis("alpha", [1e-2])

    bench.run_all_benchmarks(sys.argv)
