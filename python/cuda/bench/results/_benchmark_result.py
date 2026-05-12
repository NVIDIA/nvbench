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

from __future__ import annotations

import array
import json
import os
import sys
from collections.abc import ItemsView, Iterator, KeysView, ValuesView
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

__all__ = [
    "BenchmarkResult",
    "BenchmarkResultDevice",
    "BenchmarkResultSummary",
    "SubBenchmarkResult",
    "SubBenchmarkState",
]

ResultT = TypeVar("ResultT")
BenchmarkResultT = TypeVar("BenchmarkResultT", bound="BenchmarkResult")
_SummaryValue = int | float | str


@dataclass(frozen=True)
class BenchmarkResultDevice:
    """Device metadata parsed from an NVBench JSON result file."""

    id: int
    name: str
    data: dict[str, Any]


def read_json(filename: str | os.PathLike[str]) -> dict:
    with open(filename, "r", encoding="utf-8") as f:
        file_root = json.load(f)
    return file_root


def extract_filename(summary: dict) -> str:
    summary_data = summary["data"]
    value_data = next(filter(lambda v: v["name"] == "filename", summary_data))
    assert value_data["type"] == "string"
    return value_data["value"]


def extract_size(summary: dict) -> int:
    summary_data = summary["data"]
    value_data = next(filter(lambda v: v["name"] == "size", summary_data))
    assert value_data["type"] == "int64"
    return int(value_data["value"])


def parse_summary_value(value_data: dict) -> _SummaryValue:
    value_type = value_data["type"]
    value = value_data["value"]
    if value_type == "int64":
        return int(value)
    if value_type == "float64":
        return float(value)
    if value_type == "string":
        return value
    raise ValueError(f"unsupported summary value type: {value_type}")


@dataclass(frozen=True)
class BenchmarkResultSummary:
    """Summary record parsed from one NVBench benchmark state."""

    tag: str
    name: str | None
    hint: str | None
    hide: str | None
    description: str | None
    data: dict[str, _SummaryValue]

    @property
    def value(self) -> _SummaryValue | None:
        return self.data.get("value")

    def __getitem__(self, key: str) -> _SummaryValue:
        return self.data[key]

    def get(
        self, key: str, default: _SummaryValue | None = None
    ) -> _SummaryValue | None:
        return self.data.get(key, default)


def parse_summary(summary: dict) -> BenchmarkResultSummary:
    data = {
        value_data["name"]: parse_summary_value(value_data)
        for value_data in summary.get("data", [])
    }
    return BenchmarkResultSummary(
        tag=summary["tag"],
        name=summary.get("name"),
        hint=summary.get("hint"),
        hide=summary.get("hide"),
        description=summary.get("description"),
        data=data,
    )


def get_state_summaries(state: dict) -> list[dict]:
    return state.get("summaries") or []


def parse_summaries(state: dict) -> dict[str, BenchmarkResultSummary]:
    return {
        summary["tag"]: parse_summary(summary) for summary in get_state_summaries(state)
    }


def parse_binary_meta(state: dict, tag: str) -> tuple[int | None, str | None]:
    summaries = get_state_summaries(state)
    if not summaries:
        return None, None

    summary = next(
        filter(lambda s: s["tag"] == tag, summaries),
        None,
    )
    if not summary:
        return None, None

    sample_filename = extract_filename(summary)
    sample_count = extract_size(summary)
    return sample_count, sample_filename


def parse_samples_meta(state: dict) -> tuple[int | None, str | None]:
    return parse_binary_meta(state, "nv/json/bin:nv/cold/sample_times")


def parse_frequencies_meta(state: dict) -> tuple[int | None, str | None]:
    return parse_binary_meta(state, "nv/json/freqs-bin:nv/cold/sample_freqs")


def resolve_binary_filename(json_dir: str, binary_filename: str) -> str:
    if os.path.isabs(binary_filename):
        return binary_filename

    json_relative_filename = os.path.join(json_dir, binary_filename)
    if os.path.exists(json_relative_filename):
        return json_relative_filename

    parent_relative_filename = os.path.join(os.path.dirname(json_dir), binary_filename)
    if os.path.exists(parent_relative_filename):
        return parent_relative_filename

    if os.path.exists(binary_filename):
        return binary_filename

    return json_relative_filename


def parse_float32_binary(
    count: int | None, filename: str | None, json_dir: str
) -> array.array | None:
    if count is None or filename is None:
        return None

    values = array.array("f")
    if values.itemsize != 4:
        raise RuntimeError("array('f') is not a 32-bit float on this platform")

    filename = resolve_binary_filename(json_dir, filename)
    try:
        with open(filename, "rb") as f:
            size = os.fstat(f.fileno()).st_size
            if size % values.itemsize:
                raise ValueError("file size is not a multiple of float size")

            values.fromfile(f, size // values.itemsize)
    except FileNotFoundError:
        return None

    # Match np.fromfile(fn, "<f4"): little-endian float32.
    if sys.byteorder != "little":
        values.byteswap()

    if count != len(values):
        raise ValueError(f"expected {count} values in {filename}, found {len(values)}")
    return values


def parse_samples(state: dict, json_dir: str) -> array.array | None:
    """Return the state's sample times, or None if sample data is unavailable."""
    sample_count, samples_filename = parse_samples_meta(state)
    return parse_float32_binary(sample_count, samples_filename, json_dir)


def parse_frequencies(state: dict, json_dir: str) -> array.array | None:
    """Return the state's sample frequencies, or None if data is unavailable."""
    frequency_count, frequencies_filename = parse_frequencies_meta(state)
    return parse_float32_binary(frequency_count, frequencies_filename, json_dir)


def parse_bw(summaries: dict[str, BenchmarkResultSummary]) -> float | None:
    bwutil = summaries.get("nv/cold/bw/global/utilization")
    if bwutil is None or bwutil.value is None:
        return None

    return float(bwutil.value)


def get_axis_name(axis: dict) -> str:
    name = axis["name"]
    if af := axis.get("flags"):
        name = name + f"[{af}]"
    return name


class SubBenchmarkState:
    """Result data for one executed state of an NVBench benchmark."""

    def __init__(self, state: dict, axes_names: dict, axes_values: dict, json_dir: str):
        self.state_name = state["name"]
        self.device = state.get("device")
        self.type_config_index = state.get("type_config_index")
        self.axis_values = state.get("axis_values") or []
        self.is_skipped = state.get("is_skipped", False)
        self.skip_reason = state.get("skip_reason")
        self.summaries = parse_summaries(state)
        self.samples = parse_samples(state, json_dir)
        self.frequencies = parse_frequencies(state, json_dir)
        if (
            self.samples is not None
            and self.frequencies is not None
            and len(self.samples) != len(self.frequencies)
        ):
            raise ValueError(
                f"sample count ({len(self.samples)}) does not match "
                f"frequency count ({len(self.frequencies)})"
            )
        self.bw = parse_bw(self.summaries)

        self.point = {}
        for axis in self.axis_values:
            axis_name = axis["name"]
            name = axes_names[axis_name]
            axis_value_map = axes_values[axis_name]
            if "value" in axis:
                key = str(axis["value"])
                value = axis_value_map.get(key, key)
            else:
                input_string = axis.get("input_string")
                value = (
                    axis_value_map.get(input_string, input_string)
                    if input_string is not None
                    else ""
                )
            self.point[name] = value

    def __repr__(self) -> str:
        return str(self.__dict__)

    def name(self) -> str:
        if not self.point:
            return self.state_name
        return " ".join(f"{k}={v}" for k, v in self.point.items())

    def center(self, estimator: Callable[[array.array], ResultT]) -> ResultT | None:
        if self.samples is None:
            return None
        return estimator(self.samples)

    def center_with_frequencies(
        self, estimator: Callable[[array.array, array.array], ResultT]
    ) -> ResultT | None:
        if self.samples is None or self.frequencies is None:
            return None
        return estimator(self.samples, self.frequencies)


class SubBenchmarkResult:
    """Result data for one NVBench benchmark and its executed states."""

    def __init__(self, bench: dict, json_dir: str):
        self.name = bench["name"]
        self.devices = bench.get("devices") or []
        self.axes = bench.get("axes") or []

        axes_names = {}
        axes_values = {}
        for axis in self.axes:
            short_name = axis["name"]
            full_name = get_axis_name(axis)
            this_axis_values = {}
            for value in axis["values"]:
                input_string = value["input_string"]
                this_axis_values[input_string] = input_string
                if "value" in value:
                    this_axis_values[str(value["value"])] = input_string
            axes_names[short_name] = full_name
            axes_values[short_name] = this_axis_values

        self.states = []
        for state in bench["states"]:
            if not state.get("is_skipped", False):
                self.states.append(
                    SubBenchmarkState(state, axes_names, axes_values, json_dir)
                )

    def __repr__(self) -> str:
        return str(self.__dict__)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(
        self, state_index: int | slice
    ) -> SubBenchmarkState | list[SubBenchmarkState]:
        return self.states[state_index]

    def __iter__(self) -> Iterator[SubBenchmarkState]:
        return iter(self.states)

    def centers(
        self, estimator: Callable[[array.array], ResultT]
    ) -> dict[str, ResultT | None]:
        result = {}
        for state in self.states:
            result[state.name()] = state.center(estimator)
        return result

    def centers_with_frequencies(
        self, estimator: Callable[[array.array, array.array], ResultT]
    ) -> dict[str, ResultT | None]:
        result = {}
        for state in self.states:
            result[state.name()] = state.center_with_frequencies(estimator)
        return result


class BenchmarkResult:
    """Container for benchmark result data parsed from NVBench JSON output.

    Instances are created with :meth:`from_json` or :meth:`empty`. Direct
    construction is intentionally disabled to keep creation paths explicit.
    """

    _construction_token = object()

    def __init__(
        self,
        token=None,
    ):
        """Initialize an instance created by a BenchmarkResult class method.

        Users should call :meth:`from_json` or :meth:`empty` instead. The token
        argument is an implementation detail used to prevent direct
        construction.
        """
        if token is not self._construction_token:
            raise TypeError(
                "BenchmarkResult cannot be constructed directly; "
                "use BenchmarkResult.from_json() or BenchmarkResult.empty()"
            )

        self.metadata: Any = None
        self.devices: dict[int, BenchmarkResultDevice] = {}
        self.subbenches: dict[str, SubBenchmarkResult] = {}

    @classmethod
    def empty(cls: type[BenchmarkResultT], *, metadata: Any = None) -> BenchmarkResultT:
        """Create an empty result container with optional user metadata."""
        result = cls(cls._construction_token)
        result.metadata = metadata
        return result

    @classmethod
    def from_json(
        cls: type[BenchmarkResultT],
        json_path: str | os.PathLike[str],
        *,
        metadata: Any = None,
    ) -> BenchmarkResultT:
        """Read benchmark result data from an NVBench JSON output file."""
        result = cls.empty(metadata=metadata)
        result._parse_json(json_path)
        return result

    def _parse_json(self, json_path: str | os.PathLike[str]) -> None:
        """Populate this instance from an NVBench JSON output file."""
        json_path = os.fspath(json_path)
        json_dir = os.path.dirname(os.path.abspath(json_path))
        result_json = read_json(json_path)
        self.devices = {
            int(device["id"]): BenchmarkResultDevice(
                id=int(device["id"]),
                name=device["name"],
                data=device,
            )
            for device in result_json.get("devices", [])
        }
        for bench in result_json["benchmarks"]:
            bench_name: str = bench["name"]
            self.subbenches[bench_name] = SubBenchmarkResult(bench, json_dir)

    def __repr__(self) -> str:
        return str(self.__dict__)

    def __len__(self) -> int:
        return len(self.subbenches)

    def __iter__(self) -> Iterator[str]:
        return iter(self.subbenches)

    def __contains__(self, subbench_name: object) -> bool:
        return subbench_name in self.subbenches

    def __getitem__(self, subbench_name: str) -> SubBenchmarkResult:
        return self.subbenches[subbench_name]

    def keys(self) -> KeysView[str]:
        return self.subbenches.keys()

    def values(self) -> ValuesView[SubBenchmarkResult]:
        return self.subbenches.values()

    def items(self) -> ItemsView[str, SubBenchmarkResult]:
        return self.subbenches.items()

    def centers(
        self, estimator: Callable[[array.array], ResultT]
    ) -> dict[str, dict[str, ResultT | None]]:
        result = {}
        for subbench in self.subbenches:
            result[subbench] = self.subbenches[subbench].centers(estimator)
        return result

    def centers_with_frequencies(
        self, estimator: Callable[[array.array, array.array], ResultT]
    ) -> dict[str, dict[str, ResultT | None]]:
        result = {}
        for subbench in self.subbenches:
            result[subbench] = self.subbenches[subbench].centers_with_frequencies(
                estimator
            )
        return result
