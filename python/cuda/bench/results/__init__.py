# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
