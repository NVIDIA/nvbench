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

# The embedded NVBench prefix is present in built cuda-bench wheels, not in
# source-tree or editable-test layouts. These tests monkeypatch the package
# directory seam to validate path construction and existence checks without
# depending on an installed wheel.

from pathlib import Path

import cuda.bench as bench
import cuda.bench._paths as paths
import pytest


def test_embedded_nvbench_prefix_helpers(tmp_path, monkeypatch):
    package_dir = tmp_path / "cuda" / "bench"
    package_dir.mkdir(parents=True)
    prefix = package_dir / "cu13" / "nvbench"
    include_dir = prefix / "include"
    library_dir = prefix / "lib"
    cmake_dir = library_dir / "cmake" / "nvbench"
    cmake_dir.mkdir(parents=True)
    include_dir.mkdir()

    monkeypatch.setattr(paths, "_get_package_dir", lambda: package_dir)
    monkeypatch.setattr(paths, "_get_cuda_major_version", lambda: 13)

    assert bench.get_nvbench_prefix() == prefix
    assert bench.get_nvbench_prefix(13) == prefix
    assert bench.get_nvbench_include_dir() == include_dir
    assert bench.get_nvbench_library_dir() == library_dir
    assert bench.get_nvbench_cmake_dir() == cmake_dir

    assert isinstance(bench.get_nvbench_prefix(), Path)


def test_embedded_nvbench_prefix_helper_errors(tmp_path, monkeypatch):
    package_dir = tmp_path / "cuda" / "bench"
    package_dir.mkdir(parents=True)
    monkeypatch.setattr(paths, "_get_package_dir", lambda: package_dir)

    with pytest.raises(TypeError, match="cuda_major must be an integer"):
        bench.get_nvbench_prefix("13")

    with pytest.raises(ValueError, match="Unsupported CUDA major version"):
        bench.get_nvbench_prefix(14)

    with pytest.raises(FileNotFoundError, match="CUDA 13"):
        bench.get_nvbench_prefix(13)
