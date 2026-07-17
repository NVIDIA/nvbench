# Copyright 2025-2026 NVIDIA Corporation
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

"""CUDA Kernel Benchmarking Library Python API."""

import functools
import importlib
import importlib.metadata
import warnings
from pathlib import Path

from ._decorators import axis as axis
from ._decorators import make_register as _make_register
from ._decorators import option as option

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
    "Timer",
    "run_all_benchmarks",
)

_PUBLIC_EXPORTS = (
    *_NVBENCH_EXPORTS,
    "axis",
    "get_nvbench_cmake_dir",
    "get_nvbench_include_dir",
    "get_nvbench_library_dir",
    "get_nvbench_prefix",
    "option",
    "register",
)

_NVBENCH_TEST_EXPORTS = (
    "_test_cpp_exception",
    "_test_py_exception",
)

__all__ = list(_PUBLIC_EXPORTS)

# Optional test override used by decorator tests.
_register = None

_SUPPORTED_CUDA_MAJOR_VERSIONS = (12, 13)


def _format_supported_cuda_versions():
    return ", ".join(str(major) for major in _SUPPORTED_CUDA_MAJOR_VERSIONS)


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


def _normalize_cuda_major_version(cuda_major):
    if cuda_major is None:
        cuda_major = _get_cuda_major_version()
    elif isinstance(cuda_major, bool) or not isinstance(cuda_major, int):
        raise TypeError("cuda_major must be an integer CUDA major version")

    if cuda_major not in _SUPPORTED_CUDA_MAJOR_VERSIONS:
        raise ValueError(
            f"Unsupported CUDA major version: {cuda_major}. "
            f"Supported CUDA versions: {_format_supported_cuda_versions()}."
        )
    return cuda_major


def get_nvbench_prefix(cuda_major=None):
    """Return the embedded NVBench install prefix for a CUDA major version.

    If *cuda_major* is not provided, use the same CUDA major version selected
    by cuda-bench at runtime.
    """

    cuda_major = _normalize_cuda_major_version(cuda_major)
    prefix = Path(__file__).resolve().parent / f"cu{cuda_major}" / "nvbench"
    if not prefix.is_dir():
        raise FileNotFoundError(
            "cuda-bench wheel does not contain an embedded NVBench install "
            f"prefix for CUDA {cuda_major}.x: {prefix}"
        )
    return prefix


def _get_existing_nvbench_subdirectory(cuda_major, name):
    path = get_nvbench_prefix(cuda_major) / name
    if not path.is_dir():
        raise FileNotFoundError(f"Embedded NVBench {name} directory not found: {path}")
    return path


def get_nvbench_include_dir(cuda_major=None):
    """Return the embedded NVBench include directory."""

    return _get_existing_nvbench_subdirectory(cuda_major, "include")


def get_nvbench_library_dir(cuda_major=None):
    """Return the embedded NVBench library directory."""

    return _get_existing_nvbench_subdirectory(cuda_major, "lib")


def get_nvbench_cmake_dir(cuda_major=None):
    """Return the embedded NVBench CMake package directory."""

    path = get_nvbench_library_dir(cuda_major) / "cmake" / "nvbench"
    if not path.is_dir():
        raise FileNotFoundError(f"Embedded NVBench CMake directory not found: {path}")
    return path


def _bind_nvbench_module(module):
    for name in _NVBENCH_EXPORTS:
        globals()[name] = getattr(module, name)
        # Set module of exposed objects
        globals()[name].__module__ = __name__

    for name in _NVBENCH_TEST_EXPORTS:
        globals()[name] = getattr(module, name)

    # Expose the module as _nvbench for backward compatibility (e.g., for tests)
    globals()["_nvbench"] = module


@functools.lru_cache(maxsize=1)
def _load_nvbench_module():
    cuda_major = _get_cuda_major_version()
    extra_name = f"cu{cuda_major}"
    module_fullname = f"cuda.bench.{extra_name}._nvbench"

    try:
        module = importlib.import_module(module_fullname)
    except ImportError as e:
        raise ImportError(
            f"No cuda-bench extension found for CUDA {cuda_major}.x. "
            f"This wheel may not include support for your CUDA version. "
            f"Supported CUDA versions: {_format_supported_cuda_versions()}. "
            f"Original error: {e}"
        ) from e

    _bind_nvbench_module(module)
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
        | set(_PUBLIC_EXPORTS)
        | set(_NVBENCH_TEST_EXPORTS)
        | {"_nvbench"}
    )


def _get_register():
    if _register is not None:
        return _register
    return _load_nvbench_module().register


register = _make_register(_get_register)
register.__module__ = __name__
