# Copyright 2026 NVIDIA Corporation
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
import cupy


@bench.register()
@bench.axis.int64_power_of_two("Elements", [22, 24, 26])
def elementwise_square(state: bench.State):
    # Tell NVBench to time/synchronize the CuPy current stream. The launched
    # CuPy operation below uses that ambient stream rather than launch.get_stream().
    state.set_stream(cupy.cuda.get_current_stream())

    size = state.get_int64("Elements")
    x = cupy.random.randint(low=-16000, high=16000, size=size)
    y = cupy.empty_like(x)

    state.add_element_count(size)
    state.add_global_memory_reads(x.nbytes)
    state.add_global_memory_writes(y.nbytes)

    def launcher(launch: bench.Launch):
        cupy.square(x, out=y)

    state.exec(launcher, batched=False)


if __name__ == "__main__":
    bench.run_all_benchmarks(sys.argv)
