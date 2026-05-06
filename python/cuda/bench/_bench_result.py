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
from typing import Callable, SupportsFloat

__all__ = ["BenchResult", "SubBenchResult", "SubBenchState"]


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


def parse_samples_meta(state: dict) -> tuple[int | None, str | None]:
    summaries = state["summaries"]
    if not summaries:
        return None, None

    summary = next(
        filter(lambda s: s["tag"] == "nv/json/bin:nv/cold/sample_times", summaries),
        None,
    )
    if not summary:
        return None, None

    sample_filename = extract_filename(summary)
    sample_count = extract_size(summary)
    return sample_count, sample_filename


def resolve_sample_filename(json_dir: str, samples_filename: str) -> str:
    if os.path.isabs(samples_filename):
        return samples_filename
    return os.path.join(json_dir, samples_filename)


def parse_samples(state: dict, json_dir: str) -> array.array:
    """Return the state's sample times as an array of float32 values."""
    sample_count, samples_filename = parse_samples_meta(state)
    if sample_count is None or samples_filename is None:
        return array.array("f", [])

    samples = array.array("f")
    if samples.itemsize != 4:
        raise RuntimeError("array('f') is not a 32-bit float on this platform")

    samples_filename = resolve_sample_filename(json_dir, samples_filename)
    with open(samples_filename, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        if size % samples.itemsize:
            raise ValueError("file size is not a multiple of float size")

        samples.fromfile(f, size // samples.itemsize)

    # Match np.fromfile(fn, "<f4"): little-endian float32.
    if sys.byteorder != "little":
        samples.byteswap()

    if sample_count != len(samples):
        raise ValueError(
            f"expected {sample_count} samples in {samples_filename}, "
            f"found {len(samples)}"
        )
    return samples


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
        self.samples = parse_samples(state, json_dir)
        self.bw = parse_bw(state)

        self.point = {}
        for axis in state["axis_values"]:
            axis_name = axis["name"]
            name = axes_names[axis_name]
            value = axes_values[axis_name][axis["value"]]
            self.point[name] = value

    def __repr__(self) -> str:
        return str(self.__dict__)

    def name(self) -> str:
        return " ".join(f"{k}={v}" for k, v in self.point.items())

    def center(
        self, estimator: Callable[[array.array], SupportsFloat]
    ) -> SupportsFloat:
        return estimator(self.samples)


class SubBenchResult:
    def __init__(self, bench: dict, json_dir: str):
        axes_names = {}
        axes_values = {}
        for axis in bench["axes"]:
            short_name = axis["name"]
            full_name = get_axis_name(axis)
            this_axis_values = {}
            for value in axis["values"]:
                if "value" in value:
                    this_axis_values[str(value["value"])] = value["input_string"]
                else:
                    this_axis_values[value["input_string"]] = value["input_string"]
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

    def centers(
        self, estimator: Callable[[array.array], SupportsFloat]
    ) -> dict[str, SupportsFloat]:
        result = {}
        for state in self.states:
            result[state.name()] = state.center(estimator)
        return result


class BenchResult:
    """Parsed result data from an NVBench JSON output file."""

    def __init__(self, json_fn: str, *, code: int = 0, elapsed: float = 0.0):
        self.code = code
        self.elapsed = elapsed
        self.subbenches: dict[str, SubBenchResult] = {}

        if json_fn:
            if code == 0:
                json_dir = os.path.dirname(os.path.abspath(json_fn))
                for bench in read_json(json_fn)["benchmarks"]:
                    bench_name: str = bench["name"]
                    self.subbenches[bench_name] = SubBenchResult(bench, json_dir)

    def __repr__(self) -> str:
        return str(self.__dict__)

    def centers(
        self, estimator: Callable[[array.array], SupportsFloat]
    ) -> dict[str, dict[str, SupportsFloat]]:
        result = {}
        for subbench in self.subbenches:
            result[subbench] = self.subbenches[subbench].centers(estimator)
        return result
