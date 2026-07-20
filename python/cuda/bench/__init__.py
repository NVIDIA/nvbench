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

from ._decorators import axis as axis
from ._decorators import make_register as _make_register
from ._decorators import option as option
from ._paths import _format_supported_cuda_versions as _format_supported_cuda_versions
from ._paths import _get_cuda_major_version as _get_cuda_major_version
from ._paths import get_nvbench_cmake_dir as get_nvbench_cmake_dir
from ._paths import get_nvbench_include_dir as get_nvbench_include_dir
from ._paths import get_nvbench_library_dir as get_nvbench_library_dir
from ._paths import get_nvbench_prefix as get_nvbench_prefix

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
