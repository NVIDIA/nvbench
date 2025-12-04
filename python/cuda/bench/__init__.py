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

import importlib
import importlib.metadata
import warnings

from cuda.pathfinder import (  # type: ignore[import-not-found]
    load_nvidia_dynamic_lib,
)

try:
    __version__ = importlib.metadata.version("pynvbench")
except Exception as e:
    __version__ = "0.0.0dev"
    warnings.warn(
        "Could not retrieve version of pynvbench package dynamically from its metadata. "
        f"Exception {e} was raised. "
        f"Version is set to fall-back value '{__version__}' instead."
    )


# Detect CUDA runtime version and load appropriate extension
def _get_cuda_major_version():
    """Detect the CUDA runtime major version."""
    try:
        from cuda import cuda as cuda_bindings

        err, version = cuda_bindings.cuRuntimeGetVersion()
        if err != cuda_bindings.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Failed to get CUDA runtime version: {err}")
        # Version is encoded as (major * 1000) + (minor * 10)
        major = version // 1000
        return major
    except ImportError:
        raise ImportError(
            "cuda-bindings is required for runtime CUDA version detection. "
            "Install with: pip install pynvbench[cu12] or pip install pynvbench[cu13]"
        )


_cuda_major = _get_cuda_major_version()
_extension_name = f"_nvbench_cu{_cuda_major}"

try:
    _nvbench_module = importlib.import_module(f"cuda.bench.{_extension_name}")
except ImportError as e:
    raise ImportError(
        f"No pynvbench extension found for CUDA {_cuda_major}.x. "
        f"This wheel may not include support for your CUDA version. "
        f"Supported CUDA versions: 12, 13. "
        f"Original error: {e}"
    )

# Load required NVIDIA libraries
for libname in ("cupti", "nvperf_target", "nvperf_host"):
    load_nvidia_dynamic_lib(libname)

# Import and expose all public symbols from the CUDA-specific extension
Benchmark = _nvbench_module.Benchmark
CudaStream = _nvbench_module.CudaStream
Launch = _nvbench_module.Launch
NVBenchRuntimeError = _nvbench_module.NVBenchRuntimeError
State = _nvbench_module.State
register = _nvbench_module.register
run_all_benchmarks = _nvbench_module.run_all_benchmarks

# Clean up internal symbols
del (
    load_nvidia_dynamic_lib,
    _nvbench_module,
    _cuda_major,
    _extension_name,
    _get_cuda_major_version,
)
