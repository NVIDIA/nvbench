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

from array import array
from collections.abc import Callable, ItemsView, Iterator, KeysView, ValuesView
from os import PathLike
from typing import Any, TypeVar, overload

ResultT = TypeVar("ResultT")
BenchmarkResultT = TypeVar("BenchmarkResultT", bound="BenchmarkResult")
_SummaryValue = int | float | str

class BenchmarkResultDevice:
    id: int
    name: str
    data: dict[str, Any]

class BenchmarkResultSummary:
    tag: str
    name: str | None
    hint: str | None
    hide: str | None
    description: str | None
    data: dict[str, _SummaryValue]
    @property
    def value(self) -> _SummaryValue | None: ...
    def __getitem__(self, key: str) -> _SummaryValue: ...
    def get(
        self, key: str, default: _SummaryValue | None = None
    ) -> _SummaryValue | None: ...

class SubBenchmarkState:
    state_name: str
    device: int | None
    type_config_index: int | None
    axis_values: list[dict[str, Any]]
    is_skipped: bool
    skip_reason: str | None
    summaries: dict[str, BenchmarkResultSummary]
    samples: array | None
    frequencies: array | None
    bw: float | None
    point: dict[str, str]
    def name(self) -> str: ...
    def center(self, estimator: Callable[[array], ResultT]) -> ResultT | None: ...
    def center_with_frequencies(
        self, estimator: Callable[[array, array], ResultT]
    ) -> ResultT | None: ...

class SubBenchmarkResult:
    name: str
    devices: list[int]
    axes: list[dict[str, Any]]
    states: list[SubBenchmarkState]
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, state_index: int) -> SubBenchmarkState: ...
    @overload
    def __getitem__(self, state_index: slice) -> list[SubBenchmarkState]: ...
    def __iter__(self) -> Iterator[SubBenchmarkState]: ...
    def centers(
        self, estimator: Callable[[array], ResultT]
    ) -> dict[str, ResultT | None]: ...
    def centers_with_frequencies(
        self, estimator: Callable[[array, array], ResultT]
    ) -> dict[str, ResultT | None]: ...

class BenchmarkResult:
    metadata: Any
    devices: dict[int, BenchmarkResultDevice]
    subbenches: dict[str, SubBenchmarkResult]
    def __init__(self, token: object | None = None) -> None: ...
    @classmethod
    def empty(
        cls: type[BenchmarkResultT], *, metadata: Any = None
    ) -> BenchmarkResultT: ...
    @classmethod
    def from_json(
        cls: type[BenchmarkResultT],
        json_path: str | PathLike[str],
        *,
        metadata: Any = None,
    ) -> BenchmarkResultT: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[str]: ...
    def __contains__(self, subbench_name: object) -> bool: ...
    def __getitem__(self, subbench_name: str) -> SubBenchmarkResult: ...
    def keys(self) -> KeysView[str]: ...
    def values(self) -> ValuesView[SubBenchmarkResult]: ...
    def items(self) -> ItemsView[str, SubBenchmarkResult]: ...
    def centers(
        self, estimator: Callable[[array], ResultT]
    ) -> dict[str, dict[str, ResultT | None]]: ...
    def centers_with_frequencies(
        self, estimator: Callable[[array, array], ResultT]
    ) -> dict[str, dict[str, ResultT | None]]: ...
