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
import cuda.cccl.cooperative.experimental as coop
import numba
import numpy as np
from numba import cuda


class BitsetRing:
    """
    Addition operation over ring fixed width unsigned integers
    with ring_plus = bitwise_or and ring_mul = bitwise_and,
         ring_zero = 0, ring_one = -1
    """

    def __init__(self):
        self.dt = np.uint64
        self.zero = self.dt(0)
        self.one = np.bitwise_invert(self.zero)

    @staticmethod
    def add(op1, op2):
        return op1 | op2

    @staticmethod
    def mul(op1, op2):
        return op1 & op2


def as_cuda_Stream(cs: bench.CudaStream) -> cuda.cudadrv.driver.Stream:
    return cuda.external_stream(cs.addressof())


def multi_block_bench(state: bench.State):
    threads_per_block = state.get_int64("ThreadsPerBlock")
    num_blocks = state.get_int64("NumBlocks")
    total_elements = threads_per_block * num_blocks

    if total_elements > 2**26:
        state.skip(reason="Memory footprint over threshold")
        return

    ring = BitsetRing()
    block_reduce = coop.block.reduce(numba.uint64, threads_per_block, BitsetRing.add)

    @cuda.jit(link=block_reduce.files)
    def kernel(inp_arr, out_arr):
        # Each thread contributes one element
        block_idx = cuda.blockIdx.x
        thread_idx = cuda.threadIdx.x
        global_idx = block_idx * threads_per_block + thread_idx

        block_output = block_reduce(inp_arr[global_idx])

        # Only thread 0 of each block writes the result
        if thread_idx == 0:
            out_arr[block_idx] = block_output

    h_inp = np.arange(1, total_elements + 1, dtype=ring.dt)
    d_inp = cuda.to_device(h_inp)
    d_out = cuda.device_array(num_blocks, dtype=ring.dt)

    state.add_element_count(total_elements)
    state.add_global_memory_reads(total_elements * h_inp.itemsize)
    state.add_global_memory_writes(num_blocks * h_inp.itemsize)

    def launcher(launch: bench.Launch):
        cuda_s = as_cuda_Stream(launch.get_stream())
        kernel[num_blocks, threads_per_block, cuda_s, 0](d_inp, d_out)

    state.exec(launcher)


if __name__ == "__main__":
    b = bench.register(multi_block_bench)
    b.add_int64_axis("ThreadsPerBlock", [64, 128, 192, 256])
    b.add_int64_power_of_two_axis("NumBlocks", [10, 11, 12, 14, 16])

    bench.run_all_benchmarks(sys.argv)
