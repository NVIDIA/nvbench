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

"""Helpers for locating files installed inside cuda-bench wheels."""

from pathlib import Path

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


# Path helpers use the installed package layout. This helper keeps that lookup
# isolated so tests do not have to monkeypatch `__file__` assuming the current
# __file__-based implementation, which is not the only possible way to locate
# package data.
def _get_package_dir():
    return Path(__file__).resolve().parent


def get_nvbench_prefix(cuda_major=None):
    """Return the embedded NVBench install prefix for a CUDA major version.

    If *cuda_major* is not provided, use the same CUDA major version selected
    by cuda-bench at runtime.
    """

    cuda_major = _normalize_cuda_major_version(cuda_major)
    prefix = _get_package_dir() / f"cu{cuda_major}" / "nvbench"
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
