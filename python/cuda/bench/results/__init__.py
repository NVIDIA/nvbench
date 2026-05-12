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

"""Utilities for reading NVBench JSON benchmark result files."""

from ._benchmark_result import (
    BenchmarkResult,
    BenchmarkResultDevice,
    BenchmarkResultSummary,
    SubBenchmarkResult,
    SubBenchmarkState,
)

BenchmarkResult.__module__ = __name__
BenchmarkResultDevice.__module__ = __name__
BenchmarkResultSummary.__module__ = __name__
SubBenchmarkResult.__module__ = __name__
SubBenchmarkState.__module__ = __name__

__all__ = [
    "BenchmarkResult",
    "BenchmarkResultDevice",
    "BenchmarkResultSummary",
    "SubBenchmarkResult",
    "SubBenchmarkState",
]
