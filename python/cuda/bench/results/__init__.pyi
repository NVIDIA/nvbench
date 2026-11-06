# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from array import array
from collections.abc import Callable, ItemsView, Iterator, KeysView, ValuesView
from os import PathLike
from typing import Any, TypeAlias, TypeVar, overload

_ResultT = TypeVar("_ResultT")
_BenchmarkResultT = TypeVar("_BenchmarkResultT", bound=BenchmarkResult)
_SummaryValue: TypeAlias = int | float | str | None

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
    def center(self, estimator: Callable[[array], _ResultT]) -> _ResultT | None: ...
    def center_with_frequencies(
        self, estimator: Callable[[array, array], _ResultT]
    ) -> _ResultT | None: ...

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
        self, estimator: Callable[[array], _ResultT]
    ) -> dict[str, _ResultT | None]: ...
    def centers_with_frequencies(
        self, estimator: Callable[[array, array], _ResultT]
    ) -> dict[str, _ResultT | None]: ...

class BenchmarkResult:
    metadata: Any
    devices: dict[int, BenchmarkResultDevice]
    subbenches: dict[str, SubBenchmarkResult]
    def __init__(self, token: object | None = None) -> None: ...
    @classmethod
    def empty(
        cls: type[_BenchmarkResultT], *, metadata: Any = None
    ) -> _BenchmarkResultT: ...
    @classmethod
    def from_json(
        cls: type[_BenchmarkResultT],
        json_path: str | PathLike[str],
        *,
        metadata: Any = None,
    ) -> _BenchmarkResultT: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[str]: ...
    def __contains__(self, subbench_name: object) -> bool: ...
    def __getitem__(self, subbench_name: str) -> SubBenchmarkResult: ...
    def keys(self) -> KeysView[str]: ...
    def values(self) -> ValuesView[SubBenchmarkResult]: ...
    def items(self) -> ItemsView[str, SubBenchmarkResult]: ...
    def centers(
        self, estimator: Callable[[array], _ResultT]
    ) -> dict[str, dict[str, _ResultT | None]]: ...
    def centers_with_frequencies(
        self, estimator: Callable[[array, array], _ResultT]
    ) -> dict[str, dict[str, _ResultT | None]]: ...
