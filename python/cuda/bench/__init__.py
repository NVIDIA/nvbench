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

try:
    __version__ = importlib.metadata.version("cuda-bench")
except Exception as e:
    __version__ = "0.0.0dev"
    warnings.warn(
        "Could not retrieve version of cuda-bench package dynamically from its metadata. "
        f"Exception {e} was raised. "
        f"Version is set to fall-back value '{__version__}' instead."
    )


# Detect CUDA runtime version and load appropriate extension
def _get_cuda_major_version():
    """Detect the CUDA runtime major version."""
    try:
        import cuda.bindings

        # Get CUDA version from cuda-bindings package version
        # cuda-bindings version is in format like "12.9.1" or "13.0.0"
        version_str = cuda.bindings.__version__
        major = int(version_str.split(".")[0])
        return major
    except ImportError:
        raise ImportError(
            "cuda-bindings is required for runtime CUDA version detection. "
            "Install with: pip install cuda-bench[cu12] or pip install cuda-bench[cu13]"
        )


_cuda_major = _get_cuda_major_version()
_extra_name = f"cu{_cuda_major}"
_module_fullname = f"cuda.bench.{_extra_name}._nvbench"

try:
    _nvbench_module = importlib.import_module(_module_fullname)
except ImportError as e:
    raise ImportError(
        f"No cuda-bench extension found for CUDA {_cuda_major}.x. "
        f"This wheel may not include support for your CUDA version. "
        f"Supported CUDA versions: 12, 13. "
        f"Original error: {e}"
    )

# Import and expose all public symbols from the CUDA-specific extension
Benchmark = _nvbench_module.Benchmark
CudaStream = _nvbench_module.CudaStream
Launch = _nvbench_module.Launch
NVBenchRuntimeError = _nvbench_module.NVBenchRuntimeError
State = _nvbench_module.State
register = _nvbench_module.register
run_all_benchmarks = _nvbench_module.run_all_benchmarks
test_cpp_exception = _nvbench_module.test_cpp_exception
test_py_exception = _nvbench_module.test_py_exception

# Expose the module as _nvbench for backward compatibility (e.g., for tests)
_nvbench = _nvbench_module

# Clean up internal symbols
del (
    _nvbench_module,
    _cuda_major,
    _extra_name,
    _module_fullname,
    _get_cuda_major_version,
)

__doc__ = """
CUDA Kernel Benchmarking Library Python API
"""
