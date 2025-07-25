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

# ============================================
# PLEASE KEEP IN SYNC WITH py_nvbench.cpp FILE
# ============================================
# Please be sure to keep these type hints and docstring in sync
# with the pybind11 bindings in ``../../src/py_nvbench.cpp``

# Use mypy's stubgen to auto-generate stubs using
# ``stubgen -m cuda.nvbench._nvbench`` and compare
# stubs in generated out/cuda/nvbench/_nvbench.pyi
# with definitions given here.

from collections.abc import Callable, Sequence
from typing import Optional, Self, Union

class CudaStream:
    """Represents CUDA stream

    Note
    ----
        The class is not directly constructible.
    """
    def __cuda_stream__(self) -> tuple[int]:
        """
        Special method implement CUDA stream protocol
        from `cuda.core`. Returns a pair of integers:
        (protocol_version, integral_value_of_cudaStream_t pointer)

        Example
        -------
            import cuda.core.experimental as core
            import cuda.nvbench as nvbench

            def bench(state: nvbench.State):
                dev = core.Device(state.get_device())
                dev.set_current()
                # converts CudaString to core.Stream
                # using __cuda_stream__ protocol
                dev.create_stream(state.get_stream())
        """
        ...

    def addressof(self) -> int:
        "Integral value of address of driver's CUDA stream struct"
        ...

class Benchmark:
    """Represents NVBench benchmark.

    Note
    ----
        The class is not user-constructible.
        Use `~register` function to create Benchmark and register
        it with NVBench.
    """
    def get_name(self) -> str:
        "Get benchmark name"
        ...
    def add_int64_axis(self, name: str, values: Sequence[int]) -> Self:
        "Add integral type parameter axis with given name and values to sweep over"
        ...
    def add_int64_power_of_two_axis(self, name: str, values: Sequence[int]) -> Self:
        "Add integral type parameter axis with given name and values to sweep over"
        ...
    def add_float64_axis(self, name: str, values: Sequence[float]) -> Self:
        "Add floating-point type parameter axis with given name and values to sweep over"
        ...
    def add_string_axis(sef, name: str, values: Sequence[str]) -> Self:
        "Add string type parameter axis with given name and values to sweep over"
        ...
    def set_name(self, name: str) -> Self:
        ""
        ...
    def set_is_cpu_only(self, is_cpu_only: bool) -> Self:
        "Set whether this benchmark only executes on CPU"
        ...
    def set_run_once(self, v: bool) -> Self:
        "Set whether all benchmark configurations are executed only once"
        ...

class Launch:
    """Configuration object for function launch.

    Note
    ----
        The class is not user-constructible.
    """
    def get_stream(self) -> CudaStream:
        "Get CUDA stream of this configuration"
        ...

class State:
    """Represent benchmark configuration state.

    Note
    ----
        The class is not user-constructible.
    """
    def has_device(self) -> bool:
        "True if configuration has a device"
        ...
    def has_printers(self) -> bool:
        "True if configuration has a printer"
        ...
    def get_device(self) -> Union[int, None]:
        "Get device_id of the device from this configuration"
        ...
    def get_stream(self) -> CudaStream:
        "CudaStream object from this configuration"
        ...
    def get_int64(self, name: str) -> int:
        "Get value for given Int64 axis from this configuration"
        ...
    def get_int64_or_default(self, name: str, default_value: int) -> int:
        "Get value for given Int64 axis from this configuration"
        ...
    def get_float64(self, name: str) -> float:
        "Get value for given Float64 axis from this configuration"
        ...
    def get_float64_or_default(self, name: str, default_value: float) -> float:
        "Get value for given Float64 axis from this configuration"
        ...
    def get_string(self, name: str) -> str:
        "Get value for given String axis from this configuration"
        ...
    def get_string_or_default(self, name: str, default_value: str) -> str:
        "Get value for given String axis from this configuration"
        ...
    def add_element_count(self, count: int, column_name: Optional[str] = None) -> None:
        "Add element count"
        ...
    def set_element_count(self, count: int) -> None:
        "Set element count"
        ...
    def get_element_count(self) -> int:
        "Get element count"
        ...
    def skip(self, reason: str) -> None:
        "Skip this configuration"
        ...
    def is_skipped(self) -> bool:
        "Has this configuration been skipped"
        ...
    def get_skip_reason(self) -> str:
        "Get reason provided for skipping this configuration"
        ...
    def add_global_memory_reads(self, nbytes: int, /, column_name: str = "") -> None:
        "Inform NVBench that given amount of bytes is being read by the benchmark from global memory"
        ...
    def add_global_memory_writes(self, nbytes: int, /, column_name: str = "") -> None:
        "Inform NVBench that given amount of bytes is being written by the benchmark into global memory"
        ...
    def get_benchmark(self) -> Benchmark:
        "Get Benchmark this configuration is a part of"
        ...
    def get_throttle_threshold(self) -> float:
        "Get throttle threshold value"
        ...
    def get_min_samples(self) -> int:
        "Get the number of benchmark timings NVBench performs before stopping criterion begins being used"
        ...
    def set_min_samples(self, min_samples_count: int) -> None:
        "Set the number of benchmark timings for NVBench to perform before stopping criterion begins being used"
        ...
    def get_disable_blocking_kernel(self) -> bool:
        "True if use of blocking kernel by NVBench is disabled, False otherwise"
        ...
    def set_disable_blocking_kernel(self, flag: bool) -> None:
        "Use flag = True to disable use of blocking kernel by NVBench"
        ...
    def get_run_once(self) -> bool:
        "Boolean flag whether configuration should only run once"
        ...
    def set_run_once(self, run_once_flag: bool) -> None:
        "Set run-once flag for this configuration"
        ...
    def get_timeout(self) -> float:
        "Get time-out value for benchmark execution of this configuration"
        ...
    def set_timeout(self, duration: float) -> None:
        "Set time-out value for benchmark execution of this configuration, in seconds"
        ...
    def get_blocking_kernel_timeout(self) -> float:
        "Get time-out value for execution of blocking kernel"
        ...
    def set_blocking_kernel_timeout(self, duration: float) -> None:
        "Set time-out value for execution of blocking kernel, in seconds"
        ...
    def collect_cupti_metrics(self) -> None:
        "Request NVBench to record CUPTI metrics while running benchmark for this configuration"
        ...
    def is_cupti_required(self) -> bool:
        "True if (some) CUPTI metrics are being collected"
        ...
    def exec(
        self,
        fn: Callable[[Launch], None],
        /,
        *,
        batched: Optional[bool] = True,
        sync: Optional[bool] = False,
    ):
        """Execute callable running the benchmark.

        The callable may be executed multiple times.

        Parameters
        ----------
        fn: Callable
            Python callable with signature fn(Launch) -> None that executes the benchmark.
        batched: bool, optional
            If `True`, no cache flushing is performed between callable invocations.
            Default: `True`.
        sync: bool, optional
            True value indicates that callable performs device synchronization.
            NVBench disables use of blocking kernel in this case.
            Default: `False`.
        """
        ...
    def get_short_description(self) -> str:
        "Get short description for this configuration"
        ...
    def add_summary(self, column_name: str, value: Union[int, float, str]) -> None:
        "Add summary column with a value"
        ...
    def get_axis_values(self) -> dict[str, int | float | str]:
        "Get dictionary with axis values for this configuration"
        ...
    def get_axis_values_as_string(self) -> str:
        "Get string of space-separated name=value pairs for this configuration"
        ...

def register(fn: Callable[[State], None]) -> Benchmark:
    """
    Register bencharking function with NVBench.
    """
    ...

def run_all_benchmarks(argv: Sequence[str]) -> None:
    """
    Run all benchmarks registered with NVBench.

    Parameters
    ----------
    argv: List[str]
        Sequence of CLI arguments controlling NVBench. Usually, it is `sys.argv`.
    """
    ...

class NVBenchRuntimeError(RuntimeError):
    """An exception raised if running benchmarks encounters an error"""

    ...
