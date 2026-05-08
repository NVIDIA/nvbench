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

import array
import json
import os
import sys
from collections.abc import Iterator
from typing import Any, Callable, TypeVar

__all__ = ["BenchResult", "SubBenchResult", "SubBenchState"]

ResultT = TypeVar("ResultT")
_SummaryValue = int | float | str
_SummaryData = _SummaryValue | dict[str, _SummaryValue]


def read_json(filename: str) -> dict:
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


def extract_bw(summary: dict) -> float:
    summary_data = summary["data"]
    value_data = next(filter(lambda v: v["name"] == "value", summary_data))
    assert value_data["type"] == "float64"
    return float(value_data["value"])


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


def parse_summary_data(summary: dict) -> _SummaryData:
    summary_values = {
        value_data["name"]: parse_summary_value(value_data)
        for value_data in summary["data"]
    }
    if len(summary_values) == 1 and "value" in summary_values:
        return summary_values["value"]
    return summary_values


def parse_summaries(state: dict) -> dict[str, _SummaryData]:
    return {
        summary["tag"]: parse_summary_data(summary) for summary in state["summaries"]
    }


def parse_binary_meta(state: dict, tag: str) -> tuple[int | None, str | None]:
    summaries = state["summaries"]
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


def parse_bw(state: dict) -> float | None:
    bwutil = next(
        filter(
            lambda s: s["tag"] == "nv/cold/bw/global/utilization", state["summaries"]
        ),
        None,
    )
    if not bwutil:
        return None

    return extract_bw(bwutil)


def get_axis_name(axis: dict) -> str:
    name = axis["name"]
    if af := axis.get("flags"):
        name = name + f"[{af}]"
    return name


class SubBenchState:
    def __init__(self, state: dict, axes_names: dict, axes_values: dict, json_dir: str):
        self.state_name = state["name"]
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
        self.bw = parse_bw(state)

        self.point = {}
        for axis in state["axis_values"] or []:
            axis_name = axis["name"]
            name = axes_names[axis_name]
            value = axes_values[axis_name][axis["value"]]
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


class SubBenchResult:
    def __init__(self, bench: dict, json_dir: str):
        axes_names = {}
        axes_values = {}
        for axis in bench["axes"] or []:
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
            if not state["is_skipped"]:
                self.states.append(
                    SubBenchState(state, axes_names, axes_values, json_dir)
                )

    def __repr__(self) -> str:
        return str(self.__dict__)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(
        self, state_index: int | slice
    ) -> SubBenchState | list[SubBenchState]:
        return self.states[state_index]

    def __iter__(self) -> Iterator[SubBenchState]:
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


class BenchResult:
    """Parsed result data from an NVBench JSON output file."""

    def __init__(
        self,
        json_fn: str | None = None,
        *,
        metadata: Any = None,
        parse: bool = True,
    ):
        self.metadata = metadata
        self.subbenches: dict[str, SubBenchResult] = {}

        if json_fn and parse:
            json_dir = os.path.dirname(os.path.abspath(json_fn))
            for bench in read_json(json_fn)["benchmarks"]:
                bench_name: str = bench["name"]
                self.subbenches[bench_name] = SubBenchResult(bench, json_dir)

    def __repr__(self) -> str:
        return str(self.__dict__)

    def __getitem__(self, subbench_name: str) -> SubBenchResult:
        return self.subbenches[subbench_name]

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
