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


_NVBENCH_EXPORTS = (
    "Benchmark",
    "CudaStream",
    "Launch",
    "NVBenchRuntimeError",
    "State",
    "register",
    "run_all_benchmarks",
)

_NVBENCH_TEST_EXPORTS = (
    "_test_cpp_exception",
    "_test_py_exception",
)

__all__ = [
    *_NVBENCH_EXPORTS,
]

_nvbench_module = None


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


def _bind_nvbench_module(module):
    for name in _NVBENCH_EXPORTS:
        globals()[name] = getattr(module, name)
        # Set module of exposed objects
        globals()[name].__module__ = __name__

    for name in _NVBENCH_TEST_EXPORTS:
        globals()[name] = getattr(module, name)

    # Expose the module as _nvbench for backward compatibility (e.g., for tests)
    globals()["_nvbench"] = module


def _load_nvbench_module():
    global _nvbench_module

    if _nvbench_module is not None:
        return _nvbench_module

    cuda_major = _get_cuda_major_version()
    extra_name = f"cu{cuda_major}"
    module_fullname = f"cuda.bench.{extra_name}._nvbench"

    try:
        module = importlib.import_module(module_fullname)
    except ImportError as e:
        raise ImportError(
            f"No cuda-bench extension found for CUDA {cuda_major}.x. "
            f"This wheel may not include support for your CUDA version. "
            f"Supported CUDA versions: 12, 13. "
            f"Original error: {e}"
        ) from e

    _bind_nvbench_module(module)
    _nvbench_module = module
    return module


def __getattr__(name):
    if name == "_nvbench":
        return _load_nvbench_module()

    if name in _NVBENCH_EXPORTS + _NVBENCH_TEST_EXPORTS:
        _load_nvbench_module()
        return globals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(
        set(globals())
        | set(_NVBENCH_EXPORTS)
        | set(_NVBENCH_TEST_EXPORTS)
        | {"_nvbench"}
    )


__doc__ = """
CUDA Kernel Benchmarking Library Python API
"""
