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

import cuda.nvbench as nvbench
import numpy as np
from numba import cuda


def as_cuda_Stream(cs: nvbench.CudaStream) -> cuda.cudadrv.driver.Stream:
    return cuda.external_stream(cs.addressof())


def make_kernel(items_per_thread: int) -> cuda.dispatcher.CUDADispatcher:
    @cuda.jit
    def kernel(stride: np.uintp, elements: np.uintp, in_arr, out_arr):
        tid = cuda.grid(1)
        step = cuda.gridDim.x * cuda.blockDim.x
        for i in range(stride * tid, stride * elements, stride * step):
            for j in range(items_per_thread):
                read_id = (items_per_thread * i + j) % elements
                write_id = tid + j * elements
                out_arr[write_id] = in_arr[read_id]

    return kernel


def throughput_bench(state: nvbench.State) -> None:
    stride = state.get_int64("Stride")
    ipt = state.get_int64("ItemsPerThread")

    nbytes = 128 * 1024 * 1024
    elements = nbytes // np.dtype(np.int32).itemsize

    alloc_stream = as_cuda_Stream(state.get_stream())
    inp_arr = cuda.device_array(elements, dtype=np.int32, stream=alloc_stream)
    out_arr = cuda.device_array(elements * ipt, dtype=np.int32, stream=alloc_stream)

    state.add_element_count(elements, column_name="Elements")
    state.collect_cupti_metrics()

    threads_per_block = 256
    blocks_in_grid = (elements + threads_per_block - 1) // threads_per_block

    krn = make_kernel(ipt)

    # warm-up call ensures that kernel is loaded into context
    # before blocking kernel is launched. Kernel loading may cause
    # a synchronization to occur.
    krn[blocks_in_grid, threads_per_block, alloc_stream, 0](
        stride, elements, inp_arr, out_arr
    )

    def launcher(launch: nvbench.Launch):
        exec_stream = as_cuda_Stream(launch.get_stream())
        krn[blocks_in_grid, threads_per_block, exec_stream, 0](
            stride, elements, inp_arr, out_arr
        )

    state.exec(launcher)


if __name__ == "__main__":
    b = nvbench.register(throughput_bench)
    b.add_int64_axis("Stride", [1, 2, 4])
    b.add_int64_axis("ItemsPerThread", [1, 2, 3, 4])

    nvbench.run_all_benchmarks(sys.argv)
