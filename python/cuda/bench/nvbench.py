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

"""Command-line access to the NVBench installation embedded in cuda-bench.

The cuda-bench wheel includes an NVBench CMake installation under a
CUDA-versioned package directory, such as ``cuda/bench/cu12/nvbench`` or
``cuda/bench/cu13/nvbench``. This module prints those paths for build systems,
shell scripts, and examples that need to compile native NVBench benchmarks
against the copy shipped with the Python package.

The implementation intentionally uses pure-Python path helpers. Running
``python -m cuda.bench.nvbench --prefix`` does not import the ``_nvbench``
native extension.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from . import _paths

PathGetter = Callable[[int | None], Path]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cuda.bench.nvbench",
        description="Print paths for the NVBench installation embedded in cuda-bench.",
    )
    parser.add_argument(
        "--cuda-major",
        type=int,
        default=None,
        help="CUDA major version to query. Defaults to the runtime CUDA version.",
    )

    paths = parser.add_mutually_exclusive_group(required=True)
    paths.add_argument(
        "--prefix",
        action="store_const",
        const=_paths.get_nvbench_prefix,
        dest="path_getter",
        help="Print the embedded NVBench install prefix.",
    )
    paths.add_argument(
        "--include-dir",
        action="store_const",
        const=_paths.get_nvbench_include_dir,
        dest="path_getter",
        help="Print the embedded NVBench include directory.",
    )
    paths.add_argument(
        "--library-dir",
        action="store_const",
        const=_paths.get_nvbench_library_dir,
        dest="path_getter",
        help="Print the embedded NVBench library directory.",
    )
    paths.add_argument(
        "--cmake-dir",
        action="store_const",
        const=_paths.get_nvbench_cmake_dir,
        dest="path_getter",
        help="Print the embedded NVBench CMake package directory.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    path_getter: PathGetter = args.path_getter
    try:
        print(path_getter(args.cuda_major))
    except (FileNotFoundError, ImportError, TypeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
