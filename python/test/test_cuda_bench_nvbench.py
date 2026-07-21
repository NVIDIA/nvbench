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

from pathlib import Path

import cuda.bench.nvbench as nvbench_cli


def test_prints_requested_nvbench_path(capsys, monkeypatch):
    expected_paths = {
        "prefix": Path("/tmp/nvbench"),
        "include": Path("/tmp/nvbench/include"),
        "library": Path("/tmp/nvbench/lib"),
        "cmake": Path("/tmp/nvbench/lib/cmake/nvbench"),
    }

    monkeypatch.setattr(
        nvbench_cli._paths,
        "get_nvbench_prefix",
        lambda cuda_major=None: expected_paths["prefix"],
    )
    monkeypatch.setattr(
        nvbench_cli._paths,
        "get_nvbench_include_dir",
        lambda cuda_major=None: expected_paths["include"],
    )
    monkeypatch.setattr(
        nvbench_cli._paths,
        "get_nvbench_library_dir",
        lambda cuda_major=None: expected_paths["library"],
    )
    monkeypatch.setattr(
        nvbench_cli._paths,
        "get_nvbench_cmake_dir",
        lambda cuda_major=None: expected_paths["cmake"],
    )

    assert nvbench_cli.main(["--prefix"]) == 0
    assert capsys.readouterr().out == f"{expected_paths['prefix']}\n"

    assert nvbench_cli.main(["--include-dir"]) == 0
    assert capsys.readouterr().out == f"{expected_paths['include']}\n"

    assert nvbench_cli.main(["--library-dir"]) == 0
    assert capsys.readouterr().out == f"{expected_paths['library']}\n"

    assert nvbench_cli.main(["--cmake-dir"]) == 0
    assert capsys.readouterr().out == f"{expected_paths['cmake']}\n"


def test_forwards_cuda_major(capsys, monkeypatch):
    queried_cuda_major = None

    def get_nvbench_prefix(cuda_major=None):
        nonlocal queried_cuda_major
        queried_cuda_major = cuda_major
        return Path("/tmp/nvbench")

    monkeypatch.setattr(nvbench_cli._paths, "get_nvbench_prefix", get_nvbench_prefix)

    assert nvbench_cli.main(["--cuda-major", "13", "--prefix"]) == 0
    assert queried_cuda_major == 13
    assert capsys.readouterr().out == "/tmp/nvbench\n"


def test_reports_path_errors(capsys, monkeypatch):
    def get_nvbench_prefix(cuda_major=None):
        raise FileNotFoundError("missing embedded NVBench prefix")

    monkeypatch.setattr(nvbench_cli._paths, "get_nvbench_prefix", get_nvbench_prefix)

    assert nvbench_cli.main(["--prefix"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "error: missing embedded NVBench prefix\n"
