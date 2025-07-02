# from __future__ import annotations

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
                dev = core.Device(state.getDevice())
                dev.set_current()
                # converts CudaString to core.Stream
                # using __cuda_stream__ protocol
                dev.create_stream(state.getStream())
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
    def getName(self) -> str:
        "Get benchmark name"
        ...
    def addInt64Axis(self, name: str, values: Sequence[int]) -> Self:
        "Add integral type parameter axis with given name and values to sweep over"
        ...
    def addFloat64Axis(self, name: str, values: Sequence[float]) -> Self:
        "Add floating-point type parameter axis with given name and values to sweep over"
        ...
    def addStringAxis(sef, name: str, values: Sequence[str]) -> Self:
        "Add string type parameter axis with given name and values to sweep over"
        ...

class Launch:
    """Configuration object for function launch.

    Note
    ----
        The class is not user-constructible.
    """
    def getStream(self) -> CudaStream:
        "Get CUDA stream of this configuration"
        ...

class State:
    """Represent benchmark configuration state.

    Note
    ----
        The class is not user-constructible.
    """
    def hasDevice(self) -> bool:
        "True if configuration has a device"
        ...
    def hasPrinters(self) -> bool:
        "True if configuration has a printer"
        ...
    def getDevice(self) -> Union[int, None]:
        "Get device_id of the device from this configuration"
        ...
    def getStream(self) -> CudaStream:
        "CudaStream object from this configuration"
        ...
    def getInt64(self, name: str, default_value: Optional[int] = None) -> int:
        "Get value for given Int64 axis from this configuration"
        ...
    def getFloat64(self, name: str, default_value: Optional[float] = None) -> float:
        "Get value for given Float64 axis from this configuration"
        ...
    def getString(self, name: str, default_value: Optional[str] = None) -> str:
        "Get value for given String axis from this configuration"
        ...
    def addElementCount(self, count: int, column_name: Optional[str] = None) -> None:
        "Add element count"
        ...
    def setElementCount(self, count: int) -> None:
        "Set element count"
        ...
    def getElementCount(self) -> int:
        "Get element count"
        ...
    def skip(self, reason: str) -> None:
        "Skip this configuration"
        ...
    def isSkipped(self) -> bool:
        "Has this configuration been skipped"
        ...
    def getSkipReason(self) -> str:
        "Get reason provided for skipping this configuration"
        ...
    def addGlobalMemoryReads(self, nbytes: int) -> None:
        "Inform NVBench that given amount of bytes is being read by the benchmark from global memory"
        ...
    def addGlobalMemoryWrites(self, nbytes: int) -> None:
        "Inform NVBench that given amount of bytes is being written by the benchmark into global memory"
        ...
    def getBenchmark(self) -> Benchmark:
        "Get Benchmark this configuration is a part of"
        ...
    def getThrottleThreshold(self) -> float:
        "Get throttle threshold value"
        ...
    def getMinSamples(self) -> int:
        "Get the number of benchmark timings NVBench performs before stopping criterion begins being used"
        ...
    def setMinSamples(self, count: int) -> None:
        "Set the number of benchmark timings for NVBench to perform before stopping criterion begins being used"
        ...
    def getDisableBlockingKernel(self) -> bool:
        "True if use of blocking kernel by NVBench is disabled, False otherwise"
        ...
    def setDisableBlockingKernel(self, flag: bool) -> None:
        "Use flag = True to disable use of blocking kernel by NVBench"
        ...
    def getRunOnce(self) -> bool:
        "Boolean flag whether configuration should only run once"
        ...

    def setRunOnce(self, flag: bool) -> None:
        "Set run-once flag for this configuration"
        ...
    def getTimeout(self) -> float:
        "Get time-out value for benchmark execution of this configuration"
        ...
    def setTimeout(self, duration: float) -> None:
        "Set time-out value for benchmark execution of this configuration"
        ...
    def getBlockingKernelTimeout(self) -> float:
        "Get time-out value for execution of blocking kernel"
        ...
    def setBlockingKernelTimeout(self, duration: float) -> None:
        "Set time-out value for execution of blocking kernel"
        ...
    def collectCUPTIMetrics(self) -> None:
        "Request NVBench to record CUPTI metrics while running benchmark for this configuration"
        ...
    def isCUPTIRequired(self) -> bool:
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
    def add_summary(self, column_name: str, value: Union[int, float, str]) -> None:
        "Add summary column with a value"
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
