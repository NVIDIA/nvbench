#!/usr/bin/env python
#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import math
import os
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Any, Callable, Mapping

import jsondiff
import numpy as np
import tabulate
from colorama import Fore

try:
    from nvbench_json import reader
except ImportError:
    from scripts.nvbench_json import reader


# Parse version string into tuple, "x.y.z" -> (x, y, z)
def version_tuple(v):
    return tuple(map(int, (v.split("."))))


tabulate_version = version_tuple(tabulate.__version__)

GPU_TIME_MIN_TAG = "nv/cold/time/gpu/min"
GPU_TIME_MAX_TAG = "nv/cold/time/gpu/max"
GPU_TIME_MEAN_TAG = "nv/cold/time/gpu/mean"
GPU_TIME_STDEV_TAG = "nv/cold/time/gpu/stdev/absolute"
GPU_TIME_STDEV_RELATIVE_TAG = "nv/cold/time/gpu/stdev/relative"
GPU_TIME_MEDIAN_TAG = "nv/cold/time/gpu/median"
GPU_TIME_IR_TAG = "nv/cold/time/gpu/ir/absolute"
GPU_TIME_IR_RELATIVE_TAG = "nv/cold/time/gpu/ir/relative"
GPU_SM_CLOCK_RATE_MEAN_TAG = "nv/cold/sm_clock_rate/mean"
SAMPLE_TIMES_TAG = "nv/json/bin:nv/cold/sample_times"
SAMPLE_FREQUENCIES_TAG = "nv/json/freqs-bin:nv/cold/sample_freqs"

# The reader returns an object supporting the buffer protocol. Python 3.10 does
# not provide a standard Buffer type annotation.
Float32Reader = Callable[[str], object]


def read_float32_file(filename: str) -> object:
    return np.fromfile(filename, dtype="<f4")


# These dataclasses are treated as parsed value objects. frozen=True prevents
# accidental field reassignment but does not imply deep immutability.


@dataclass(frozen=True)
class Float32BinarySource:
    count: int
    filename: str
    json_dir: str
    description: str
    reader: Float32Reader = read_float32_file

    @cached_property
    def values(self) -> np.ndarray | None:
        return read_float32_binary(
            self.count, self.filename, self.json_dir, self.description, self.reader
        )


@dataclass(frozen=True)
class GpuTimingData:
    minimum: float | None
    maximum: float | None
    mean: float | None
    stdev: float | None
    stdev_relative: float | None
    median: float | None
    interquartile_range: float | None
    interquartile_range_relative: float | None
    sm_clock_rate_mean: float | None = None
    sample_source: Float32BinarySource | None = None
    frequency_source: Float32BinarySource | None = None

    @cached_property
    def samples(self) -> np.ndarray | None:
        if self.sample_source is None:
            return None
        return self.sample_source.values

    @cached_property
    def frequencies(self) -> np.ndarray | None:
        if self.frequency_source is None:
            return None
        return self.frequency_source.values


@dataclass(frozen=True)
class TimeEstimate:
    center: float | None
    relative_dispersion: float | None


class ComparisonStatus(str, Enum):
    UNKNOWN = "????"
    UNDECIDED = "UNDECIDED"
    SAME = "SAME"
    FAST = "FAST"
    SLOW = "SLOW"


@dataclass(frozen=True)
class SummaryComparison:
    ref_estimate: TimeEstimate
    cmp_estimate: TimeEstimate
    ref_time: float
    cmp_time: float
    ref_noise: float | None
    cmp_noise: float | None
    diff: float
    frac_diff: float
    max_noise: float | None
    status: ComparisonStatus


@dataclass
class ComparisonStats:
    config_count: int = 0
    pass_count: int = 0
    improvement_count: int = 0
    regression_count: int = 0
    undecided_count: int = 0
    unknown_count: int = 0

    def record(self, status: ComparisonStatus) -> None:
        self.config_count += 1
        if status == ComparisonStatus.UNKNOWN:
            self.unknown_count += 1
        elif status == ComparisonStatus.UNDECIDED:
            self.undecided_count += 1
        elif status == ComparisonStatus.SAME:
            self.pass_count += 1
        elif status == ComparisonStatus.FAST:
            self.improvement_count += 1
        else:
            self.regression_count += 1


DeviceInfo = Mapping[str, Any]


@dataclass(frozen=True)
class ComparisonRunData:
    # Device metadata fields are treated as read-only; stats is intentionally
    # mutable and accumulates counts across one comparison run.
    stats: ComparisonStats
    ref_devices: tuple[DeviceInfo, ...]
    cmp_devices: tuple[DeviceInfo, ...]


@dataclass(frozen=True)
class BenchmarkFilterScope:
    benchmark_name: str
    axis_filters: list[dict]


@dataclass(frozen=True)
class BenchmarkFilterPlan:
    global_axis_filters: list[dict]
    benchmark_scopes: list[BenchmarkFilterScope]


class OrderedBenchmarkFilterAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        actions = getattr(namespace, self.dest, None)
        actions = [] if actions is None else list(actions)
        action_kind = "axis" if option_string in {"-a", "--axis"} else "benchmark"
        actions.append((action_kind, values))
        setattr(namespace, self.dest, actions)


def state_match_key(state):
    device_prefix = f"Device={state['device']}"
    state_name = state["name"]
    if state_name == device_prefix:
        return ""
    if state_name.startswith(f"{device_prefix} "):
        return state_name[len(device_prefix) + 1 :]
    return state_name


def group_states_by_match_key(states):
    grouped = {}
    for state in states:
        grouped.setdefault(state_match_key(state), []).append(state)
    return grouped


def state_group_counts(grouped_states):
    return Counter(
        {state_name: len(states) for state_name, states in grouped_states.items()}
    )


def format_device_ids(device_ids):
    return ", ".join(str(device_id) for device_id in device_ids)


def parse_device_filter(device_arg, option_name):
    device_arg = device_arg.strip()
    if device_arg.lower() == "all":
        return None

    values = [value.strip() for value in device_arg.split(",")]
    if not all(values):
        raise ValueError(
            f"{option_name} must be 'all', a non-negative integer, "
            "or comma-separated non-negative integers"
        )

    try:
        device_ids = [int(value) for value in values]
    except ValueError as exc:
        raise ValueError(
            f"{option_name} must be 'all', a non-negative integer, "
            "or comma-separated non-negative integers"
        ) from exc
    if any(device_id < 0 for device_id in device_ids):
        raise ValueError(
            f"{option_name} must be 'all', a non-negative integer, "
            "or comma-separated non-negative integers"
        )
    return device_ids


def select_devices(all_devices, device_filter, option_name):
    if device_filter is None:
        return list(all_devices)

    devices_by_id = {device["id"]: device for device in all_devices}
    missing_ids = [
        device_id for device_id in device_filter if device_id not in devices_by_id
    ]
    if missing_ids:
        raise ValueError(
            f"{option_name} requested device id(s) not present in input: "
            f"{format_device_ids(missing_ids)}"
        )

    return [devices_by_id[device_id] for device_id in device_filter]


def resolve_benchmark_device_ids(bench, device_filter, option_name):
    if device_filter is None:
        return list(bench["devices"])

    benchmark_device_ids = set(bench["devices"])
    missing_ids = [
        device_id
        for device_id in device_filter
        if device_id not in benchmark_device_ids
    ]
    if missing_ids:
        raise ValueError(
            f"benchmark {bench['name']!r} does not contain {option_name} "
            f"device id(s): {format_device_ids(missing_ids)}"
        )

    return device_filter


def require_matching_device_sections(reference_device_filter, compare_device_filter):
    return reference_device_filter is None and compare_device_filter is None


# TODO(opavlyk): replace with Emoji(StrEnum) after EOL of Python 3.10
class Emoji(str, Enum):
    YELLOW = "\U0001f7e1"
    BLUE = "\U0001f535"
    GREEN = "\U0001f7e2"
    RED = "\U0001f534"
    NONE = ""


def colorize(msg: str, fore: Fore, emoji: Emoji, no_color: bool) -> str:
    if no_color:
        prefix = ""
        if emoji_s := emoji.value:
            prefix = f"{emoji_s} "
        return f"{prefix}{msg}"
    else:
        return f"{fore}{msg}{Fore.RESET}"


def lookup_summary(summaries, tag):
    return next((summary for summary in summaries if summary["tag"] == tag), None)


def extract_summary_data_value(summary, name, expected_type):
    summary_tag = summary.get("tag", "<unknown>")
    for value_data in summary.get("data", []):
        if value_data.get("name") != name:
            continue

        value_type = value_data.get("type")
        if value_type != expected_type:
            raise ValueError(
                f"summary {summary_tag!r} field {name!r} has type "
                f"{value_type!r}; expected {expected_type!r}"
            )
        if "value" not in value_data:
            raise ValueError(f"summary {summary_tag!r} field {name!r} is missing value")
        return value_data["value"]

    raise ValueError(f"summary {summary_tag!r} is missing field {name!r}")


def extract_summary_value(summary):
    return extract_summary_data_value(summary, "value", "float64")


def normalize_float_value(value, *, null_value=None):
    if value is None:
        return null_value
    return float(value)


def extract_summary_float(summaries, tag, *, null_value=None):
    summary = lookup_summary(summaries, tag)
    if summary is None:
        return None
    return normalize_float_value(extract_summary_value(summary), null_value=null_value)


def extract_binary_filename(summary):
    value = extract_summary_data_value(summary, "filename", "string")
    if not isinstance(value, str):
        raise ValueError(
            f"summary {summary.get('tag', '<unknown>')!r} field 'filename' "
            "value must be a string"
        )
    return value


def extract_binary_size(summary):
    value = extract_summary_data_value(summary, "size", "int64")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"summary {summary.get('tag', '<unknown>')!r} field 'size' "
            f"value {value!r} is not an int64"
        ) from exc


def extract_binary_meta(summaries, tag):
    summary = lookup_summary(summaries, tag)
    if summary is None:
        return None, None
    return extract_binary_size(summary), extract_binary_filename(summary)


def resolve_binary_filename(json_dir, binary_filename):
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


def warn_unavailable_bulk_data(description, message):
    warnings.warn(
        f"Could not use NVBench {description} data: {message}; treating it as unavailable",
        RuntimeWarning,
        stacklevel=3,
    )


def read_float32_binary(count, filename, json_dir, description, reader):
    filename = resolve_binary_filename(json_dir, filename)
    try:
        values = np.frombuffer(reader(filename), dtype="<f4")
    except (BufferError, OSError, TypeError, ValueError) as exc:
        warn_unavailable_bulk_data(description, f"failed to read {filename!r}: {exc}")
        return None

    if count != len(values):
        warn_unavailable_bulk_data(
            description,
            f"expected {count} values in {filename!r}, found {len(values)}",
        )
        return None
    return values


def extract_float32_binary_source(summaries, tag, json_dir, description, reader):
    count, filename = extract_binary_meta(summaries, tag)
    if count is None or filename is None or json_dir is None:
        return None
    if count < 0:
        warn_unavailable_bulk_data(description, f"negative value count {count}")
        return None
    return Float32BinarySource(
        count=count,
        filename=filename,
        json_dir=json_dir,
        description=description,
        reader=reader,
    )


def extract_sample_time_source(summaries, json_dir, reader):
    return extract_float32_binary_source(
        summaries, SAMPLE_TIMES_TAG, json_dir, "sample time", reader
    )


def extract_sample_frequency_source(summaries, json_dir, reader):
    return extract_float32_binary_source(
        summaries, SAMPLE_FREQUENCIES_TAG, json_dir, "sample frequency", reader
    )


def extract_gpu_timing_data(summaries, json_dir=None, float32_reader=read_float32_file):
    sample_source = extract_sample_time_source(summaries, json_dir, float32_reader)
    frequency_source = extract_sample_frequency_source(
        summaries, json_dir, float32_reader
    )
    if (
        sample_source is not None
        and frequency_source is not None
        and sample_source.count != frequency_source.count
    ):
        warn_unavailable_bulk_data(
            "paired sample time and frequency",
            f"sample count ({sample_source.count}) does not match "
            f"frequency count ({frequency_source.count})",
        )
        sample_source = None
        frequency_source = None

    return GpuTimingData(
        minimum=extract_summary_float(summaries, GPU_TIME_MIN_TAG),
        maximum=extract_summary_float(summaries, GPU_TIME_MAX_TAG),
        mean=extract_summary_float(summaries, GPU_TIME_MEAN_TAG),
        stdev=extract_summary_float(summaries, GPU_TIME_STDEV_TAG, null_value=math.inf),
        stdev_relative=extract_summary_float(
            summaries, GPU_TIME_STDEV_RELATIVE_TAG, null_value=math.inf
        ),
        median=extract_summary_float(summaries, GPU_TIME_MEDIAN_TAG),
        interquartile_range=extract_summary_float(
            summaries, GPU_TIME_IR_TAG, null_value=math.inf
        ),
        interquartile_range_relative=extract_summary_float(
            summaries, GPU_TIME_IR_RELATIVE_TAG, null_value=math.inf
        ),
        sm_clock_rate_mean=extract_summary_float(summaries, GPU_SM_CLOCK_RATE_MEAN_TAG),
        sample_source=sample_source,
        frequency_source=frequency_source,
    )


def compute_relative_dispersion(dispersion, center):
    if (
        dispersion is None
        or center is None
        or center <= 0
        or not math.isfinite(center)
        or dispersion < 0
        or math.isnan(dispersion)
    ):
        return None
    return dispersion / center


def has_robust_estimate(summary):
    return summary.median is not None and (
        summary.interquartile_range_relative is not None
        or summary.interquartile_range is not None
    )


def has_mean_estimate(summary):
    return summary.mean is not None and (
        summary.stdev_relative is not None or summary.stdev is not None
    )


def select_relative_dispersion(relative_dispersion, absolute_dispersion, center):
    if relative_dispersion is not None:
        return relative_dispersion
    return compute_relative_dispersion(absolute_dispersion, center)


def compute_common_time_estimates(ref_timing, cmp_timing):
    if has_robust_estimate(ref_timing) and has_robust_estimate(cmp_timing):
        return (
            TimeEstimate(
                center=ref_timing.median,
                relative_dispersion=select_relative_dispersion(
                    ref_timing.interquartile_range_relative,
                    ref_timing.interquartile_range,
                    ref_timing.median,
                ),
            ),
            TimeEstimate(
                center=cmp_timing.median,
                relative_dispersion=select_relative_dispersion(
                    cmp_timing.interquartile_range_relative,
                    cmp_timing.interquartile_range,
                    cmp_timing.median,
                ),
            ),
        )

    if has_mean_estimate(ref_timing) and has_mean_estimate(cmp_timing):
        return (
            TimeEstimate(
                center=ref_timing.mean,
                relative_dispersion=select_relative_dispersion(
                    ref_timing.stdev_relative, ref_timing.stdev, ref_timing.mean
                ),
            ),
            TimeEstimate(
                center=cmp_timing.mean,
                relative_dispersion=select_relative_dispersion(
                    cmp_timing.stdev_relative, cmp_timing.stdev, cmp_timing.mean
                ),
            ),
        )

    return (
        TimeEstimate(
            center=ref_timing.mean,
            relative_dispersion=compute_relative_dispersion(
                ref_timing.stdev, ref_timing.mean
            ),
        ),
        TimeEstimate(
            center=cmp_timing.mean,
            relative_dispersion=compute_relative_dispersion(
                cmp_timing.stdev, cmp_timing.mean
            ),
        ),
    )


def compare_gpu_timings(ref_timing, cmp_timing):
    ref_estimate, cmp_estimate = compute_common_time_estimates(ref_timing, cmp_timing)

    cmp_time = cmp_estimate.center
    ref_time = ref_estimate.center

    if cmp_time is None or ref_time is None:
        return None

    if not math.isfinite(cmp_time) or not math.isfinite(ref_time):
        return None

    if cmp_time <= 0.0 or ref_time <= 0.0:
        return None

    cmp_noise = cmp_estimate.relative_dispersion
    ref_noise = ref_estimate.relative_dispersion

    diff = cmp_time - ref_time
    frac_diff = diff / ref_time

    if not has_finite_noise(ref_noise) or not has_finite_noise(cmp_noise):
        max_noise = None
        status = ComparisonStatus.UNKNOWN
    else:
        max_noise = max(ref_noise, cmp_noise)
        if abs(frac_diff) <= max_noise:
            status = ComparisonStatus.SAME
        elif diff < 0:
            status = ComparisonStatus.FAST
        else:
            status = ComparisonStatus.SLOW

    return SummaryComparison(
        ref_estimate=ref_estimate,
        cmp_estimate=cmp_estimate,
        ref_time=ref_time,
        cmp_time=cmp_time,
        ref_noise=ref_noise,
        cmp_noise=cmp_noise,
        diff=diff,
        frac_diff=frac_diff,
        max_noise=max_noise,
        status=status,
    )


def find_matching_bench(needle, haystack):
    for hay in haystack:
        if hay["name"] == needle["name"]:
            return hay
    return None


def find_device_by_id(device_id, all_devices):
    for device in all_devices:
        if device["id"] == device_id:
            return device
    return None


def format_int64_axis_value(axis_name, axis_value, axes):
    axis = next(filter(lambda ax: ax["name"] == axis_name, axes))
    axis_flags = axis["flags"]
    value = int(axis_value["value"])
    if axis_flags == "pow2":
        value = math.log2(value)
        return f"2^{value:.0f}"
    return f"{value:d}"


def format_float64_axis_value(axis_name, axis_value, axes):
    return "%.5g" % float(axis_value["value"])


def format_type_axis_value(axis_name, axis_value, axes):
    return f"{axis_value['value']}"


def format_string_axis_value(axis_name, axis_value, axes):
    return f"{axis_value['value']}"


def format_axis_value(axis_name, axis_value, axes):
    axis = next(filter(lambda ax: ax["name"] == axis_name, axes))
    axis_type = axis["type"]
    if axis_type == "int64":
        return format_int64_axis_value(axis_name, axis_value, axes)
    elif axis_type == "float64":
        return format_float64_axis_value(axis_name, axis_value, axes)
    elif axis_type == "type":
        return format_type_axis_value(axis_name, axis_value, axes)
    elif axis_type == "string":
        return format_string_axis_value(axis_name, axis_value, axes)


def make_display(name: str, display_values: list[str]) -> str:
    open_bracket, close_bracket = ("[", "]") if len(display_values) > 1 else ("", "")
    joined_values = ",".join(display_values)
    return f"{name}={open_bracket}{joined_values}{close_bracket}"


def parse_axis_filters(axis_args):
    filters = []
    for axis_arg in axis_args:
        if "=" not in axis_arg:
            raise ValueError(f"Axis filter must be NAME=VALUE: {axis_arg}")
        name, value = axis_arg.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name or not value:
            raise ValueError(f"Axis filter must be NAME=VALUE: {axis_arg}")

        values = []
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            values = [
                stripped for item in inner.split(",") if (stripped := item.strip())
            ]
        else:
            values = [value]
        display_values = list(values)

        if name.endswith("[pow2]"):
            name = name[: -len("[pow2]")].strip()
            if not name:
                raise ValueError(f"Axis filter missing name before [pow2]: {axis_arg}")
            try:
                exponents = [int(v) for v in values]
            except ValueError as exc:
                raise ValueError(
                    f"Axis filter [pow2] value must be integer: {axis_arg}"
                ) from exc
            values = [str(2**exponent) for exponent in exponents]
            display_values = [f"2^{exponent}" for exponent in exponents]

        if not values:
            raise ValueError(f"Axis filter must specify at least one value: {axis_arg}")

        display = make_display(name, display_values)
        filters.append(
            {
                "name": name,
                "values": values,
                "display": display,
            }
        )
    return filters


def build_benchmark_filter_plan(filter_actions):
    global_axis_args = []
    benchmark_scopes = []
    current_scope = None

    for action_kind, action_value in filter_actions or []:
        if action_kind == "benchmark":
            current_scope = {"benchmark_name": action_value, "axis_args": []}
            benchmark_scopes.append(current_scope)
        elif current_scope is None:
            global_axis_args.append(action_value)
        else:
            current_scope["axis_args"].append(action_value)

    return BenchmarkFilterPlan(
        global_axis_filters=parse_axis_filters(global_axis_args),
        benchmark_scopes=[
            BenchmarkFilterScope(
                benchmark_name=scope["benchmark_name"],
                axis_filters=parse_axis_filters(scope["axis_args"]),
            )
            for scope in benchmark_scopes
        ],
    )


def benchmark_is_selected(benchmark_name, filter_plan):
    return not filter_plan.benchmark_scopes or any(
        scope.benchmark_name == benchmark_name for scope in filter_plan.benchmark_scopes
    )


def axis_filter_groups_for_benchmark(benchmark_name, filter_plan):
    if not filter_plan.benchmark_scopes:
        return [filter_plan.global_axis_filters]

    matching_scopes = [
        scope
        for scope in filter_plan.benchmark_scopes
        if scope.benchmark_name == benchmark_name
    ]
    return [
        filter_plan.global_axis_filters + scope.axis_filters
        for scope in matching_scopes
    ]


def matches_axis_filters(state, axis_filters):
    if not axis_filters:
        return True

    axis_values = state.get("axis_values") or []
    for axis_filter in axis_filters:
        filter_name = axis_filter["name"]
        filter_values = axis_filter["values"]
        matched = False
        for axis_value in axis_values:
            if axis_value.get("name") != filter_name:
                continue
            value = axis_value.get("value")
            if value is None:
                continue
            if str(value) in filter_values:
                matched = True
                break
        if not matched:
            return False
    return True


def matches_axis_filter_groups(state, axis_filter_groups):
    return any(
        matches_axis_filters(state, axis_filters) for axis_filters in axis_filter_groups
    )


def matching_axis_filters(state, axis_filter_groups):
    return next(
        (
            axis_filters
            for axis_filters in axis_filter_groups
            if matches_axis_filters(state, axis_filters)
        ),
        [],
    )


def format_duration(seconds):
    if seconds >= 1:
        multiplier = 1.0
        units = "s"
    elif seconds >= 1e-3:
        multiplier = 1e3
        units = "ms"
    elif seconds >= 1e-6:
        multiplier = 1e6
        units = "us"
    else:
        multiplier = 1e6
        units = "us"
    return f"{seconds * multiplier:0.3f} {units}"


def format_percentage(percentage):
    if percentage is None:
        return "n/a"
    if math.isnan(percentage):
        return "n/a"
    if math.isinf(percentage):
        return "inf"
    return f"{percentage * 100.0:0.2f}%"


def has_finite_noise(noise):
    return noise is not None and math.isfinite(noise)


def colorize_comparison_status(status, no_color):
    if status == ComparisonStatus.UNKNOWN:
        return colorize(status.value, Fore.YELLOW, Emoji.YELLOW, no_color)
    if status == ComparisonStatus.UNDECIDED:
        return colorize(status.value, Fore.YELLOW, Emoji.YELLOW, no_color)
    if status == ComparisonStatus.SAME:
        return colorize(status.value, Fore.BLUE, Emoji.BLUE, no_color)
    if status == ComparisonStatus.FAST:
        return colorize(status.value, Fore.GREEN, Emoji.GREEN, no_color)
    return colorize(status.value, Fore.RED, Emoji.RED, no_color)


def format_axis_values(axis_values, axes, axis_filters=None):
    if not axis_values:
        return ""
    filtered_names = set()
    if axis_filters:
        filtered_names = {
            axis_filter["name"]
            for axis_filter in axis_filters
            if len(axis_filter["values"]) == 1
        }
    parts = []
    for axis_value in axis_values:
        axis_name = axis_value["name"]
        if axis_name in filtered_names:
            continue
        formatted = format_axis_value(axis_name, axis_value, axes)
        parts.append(f"{axis_name}={formatted}")
    return " ".join(parts)


def plot_comparison_entries(entries, title=None, dark=False):
    if not entries:
        print("No comparison data to plot.")
        return 1

    if not os.environ.get("DISPLAY"):
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    labels, values, statuses, bench_names = map(list, zip(*entries))

    status_colors = {
        "SLOW": "red",
        "FAST": "green",
        "SAME": "blue",
    }
    colors = [status_colors.get(status, "gray") for status in statuses]

    fig_height = max(4.0, 0.3 * len(entries) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    if dark:
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("white")

    y_pos = range(len(labels))
    ax.barh(y_pos, values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_ylim(len(labels) - 0.5, -0.5)

    separator_color = "white" if dark else "gray"
    ax.axvline(0, color=separator_color, linewidth=1, alpha=0.6)
    for index in range(1, len(bench_names)):
        if bench_names[index] != bench_names[index - 1]:
            ax.axhline(index - 0.5, color=separator_color, linewidth=0.6, alpha=0.4)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))

    if title:
        ax.set_title(title)

    min_val = min(values)
    max_val = max(values)
    if min_val == max_val:
        pad = 0.05 if min_val == 0 else abs(min_val) * 0.1
        ax.set_xlim(min_val - pad, max_val + pad)
    else:
        pad = (max_val - min_val) * 0.1
        ax.set_xlim(min_val - pad, max_val + pad)

    fig.tight_layout()

    if not os.environ.get("DISPLAY"):
        output = "nvbench_compare.png"
        fig.savefig(output, dpi=150)
        print(f"Saved comparison plot to {output}")
    else:
        plt.show()
    return 0


def compare_benches(
    run_data: ComparisonRunData,
    ref_benches,
    cmp_benches,
    threshold,
    plot_along,
    plot,
    dark,
    filter_plan,
    no_color,
    reference_device_filter=None,
    compare_device_filter=None,
    ref_json_dir=None,
    cmp_json_dir=None,
):
    if plot_along:
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme()

    comparison_entries = []
    comparison_device_names = set()
    for cmp_bench in cmp_benches:
        ref_bench = find_matching_bench(cmp_bench, ref_benches)
        if not ref_bench:
            continue
        if not benchmark_is_selected(cmp_bench["name"], filter_plan):
            continue
        axis_filter_groups = axis_filter_groups_for_benchmark(
            cmp_bench["name"], filter_plan
        )

        cmp_device_ids = resolve_benchmark_device_ids(
            cmp_bench, compare_device_filter, "--compare-devices"
        )
        ref_device_ids = resolve_benchmark_device_ids(
            ref_bench, reference_device_filter, "--reference-devices"
        )
        if len(cmp_device_ids) != len(ref_device_ids):
            raise ValueError(
                f"benchmark {cmp_bench['name']!r} has {len(ref_device_ids)} "
                f"reference device(s) but {len(cmp_device_ids)} compare device(s); "
                "nvbench_compare pairs devices by position, so each compared "
                "benchmark must contain the same number of devices"
            )

        print(f"""# {cmp_bench["name"]}\n""")

        axes = cmp_bench["axes"]
        ref_states = ref_bench["states"]
        cmp_states = cmp_bench["states"]

        axes = axes if axes else []

        headers = [x["name"] for x in axes]
        colalign = ["center"] * len(headers)

        headers.append("Ref Time")
        colalign.append("right")
        headers.append("Ref Noise")
        colalign.append("right")
        headers.append("Cmp Time")
        colalign.append("right")
        headers.append("Cmp Noise")
        colalign.append("right")
        headers.append("Diff")
        colalign.append("right")
        headers.append("%Diff")
        colalign.append("right")
        headers.append("Status")
        colalign.append("center")

        for cmp_device_index, cmp_device_id in enumerate(cmp_device_ids):
            ref_device_id = ref_device_ids[cmp_device_index]
            ref_device_states = [
                state
                for state in ref_states
                if state["device"] == ref_device_id
                and matches_axis_filter_groups(state, axis_filter_groups)
            ]
            cmp_device_states = [
                state
                for state in cmp_states
                if state["device"] == cmp_device_id
                and matches_axis_filter_groups(state, axis_filter_groups)
            ]
            ref_states_by_name = group_states_by_match_key(ref_device_states)
            cmp_states_by_name = group_states_by_match_key(cmp_device_states)
            ref_state_counts = state_group_counts(ref_states_by_name)
            cmp_state_counts = state_group_counts(cmp_states_by_name)
            if ref_state_counts != cmp_state_counts:
                raise ValueError(
                    f"benchmark {cmp_bench['name']!r} device pair "
                    f"ref={ref_device_id} cmp={cmp_device_id} has mismatched "
                    f"state occurrences: ref={dict(ref_state_counts)}, "
                    f"cmp={dict(cmp_state_counts)}"
                )

            rows = []
            plot_data: dict[str, dict[str, dict[float, float | None]]] = {
                "cmp": {},
                "ref": {},
                "cmp_noise": {},
                "ref_noise": {},
            }
            counters: dict[str, int] = {}

            for cmp_state in cmp_device_states:
                cmp_state_name = state_match_key(cmp_state)
                occurrence = counters.get(cmp_state_name, 0)
                counters[cmp_state_name] = occurrence + 1
                # Duplicate state names are matched by occurrence order within
                # the filtered device section.
                ref_state = ref_states_by_name[cmp_state_name][occurrence]
                axis_values = cmp_state["axis_values"]
                if not axis_values:
                    axis_values = []

                row = []
                for axis_value in axis_values:
                    axis_value_name = axis_value["name"]
                    row.append(format_axis_value(axis_value_name, axis_value, axes))

                cmp_summaries = cmp_state["summaries"]
                ref_summaries = ref_state["summaries"]

                if not ref_summaries or not cmp_summaries:
                    continue

                # TODO: Use other timings, too. Maybe multiple rows, with a
                # "Timing" column + values "CPU/GPU/Batch"?
                cmp_gpu_time = extract_gpu_timing_data(cmp_summaries, cmp_json_dir)
                ref_gpu_time = extract_gpu_timing_data(ref_summaries, ref_json_dir)
                comparison = compare_gpu_timings(ref_gpu_time, cmp_gpu_time)
                if comparison is None:
                    continue

                if plot_along:
                    axis_name_parts = []
                    axis_value = None
                    for av in axis_values:
                        if av["name"] != plot_along:
                            axis_name_parts.append(f"""{av["name"]} = {av["value"]}""")
                        else:
                            axis_value = float(av["value"])
                    if axis_value is not None:
                        axis_name = ", ".join(axis_name_parts)

                        if axis_name not in plot_data["cmp"]:
                            plot_data["cmp"][axis_name] = {}
                            plot_data["ref"][axis_name] = {}
                            plot_data["cmp_noise"][axis_name] = {}
                            plot_data["ref_noise"][axis_name] = {}

                        plot_data["cmp"][axis_name][axis_value] = comparison.cmp_time
                        plot_data["ref"][axis_name][axis_value] = comparison.ref_time
                        plot_data["cmp_noise"][axis_name][axis_value] = (
                            comparison.cmp_noise
                        )
                        plot_data["ref_noise"][axis_name][axis_value] = (
                            comparison.ref_noise
                        )

                run_data.stats.record(comparison.status)
                status = colorize_comparison_status(comparison.status, no_color)

                if abs(comparison.frac_diff) >= threshold:
                    axis_filters = matching_axis_filters(cmp_state, axis_filter_groups)
                    row.append(format_duration(comparison.ref_time))
                    row.append(format_percentage(comparison.ref_noise))
                    row.append(format_duration(comparison.cmp_time))
                    row.append(format_percentage(comparison.cmp_noise))
                    row.append(format_duration(comparison.diff))
                    row.append(format_percentage(comparison.frac_diff))
                    row.append(status)

                    rows.append(row)
                    if plot:
                        axis_label = format_axis_values(axis_values, axes, axis_filters)
                        if axis_label:
                            label = f"""{cmp_bench["name"]} | {axis_label}"""
                        else:
                            label = cmp_bench["name"]
                        cmp_device = find_device_by_id(
                            cmp_state["device"], run_data.cmp_devices
                        )
                        if cmp_device:
                            comparison_device_names.add(cmp_device["name"])
                        comparison_entries.append(
                            (
                                label,
                                comparison.frac_diff,
                                comparison.status.value,
                                cmp_bench["name"],
                            )
                        )

            if len(rows) == 0:
                continue

            cmp_device = find_device_by_id(cmp_device_id, run_data.cmp_devices)
            ref_device = find_device_by_id(ref_device_id, run_data.ref_devices)
            if ref_device is None or cmp_device is None:
                raise ValueError(
                    f"benchmark {cmp_bench['name']!r} references device pair "
                    f"ref={ref_device_id} cmp={cmp_device_id}, but device metadata is missing"
                )

            if cmp_device == ref_device:
                print(f"## [{cmp_device['id']}] {cmp_device['name']}\n")
            else:
                print(
                    f"## [{ref_device['id']}] {ref_device['name']} vs. "
                    f"[{cmp_device['id']}] {cmp_device['name']}\n"
                )
            # colalign and github format require tabulate 0.8.3
            if tabulate_version >= (0, 8, 3):
                print(
                    tabulate.tabulate(
                        rows, headers=headers, colalign=colalign, tablefmt="github"
                    )
                )
            else:
                print(tabulate.tabulate(rows, headers=headers, tablefmt="markdown"))

            print("")

            if plot_along:
                fig = plt.figure()
                try:
                    plt.xscale("log")
                    plt.yscale("log")
                    plt.xlabel(plot_along)
                    plt.ylabel("time [s]")
                    plt.title(cmp_device["name"])

                    def plot_line(key, shape, label, data_axis, data=plot_data):
                        axis_times = data[key][data_axis]
                        if not axis_times:
                            return
                        axis_noise = data[key + "_noise"][data_axis]
                        series = sorted(
                            (
                                (
                                    float(axis_value),
                                    axis_times[axis_value],
                                    axis_noise[axis_value],
                                )
                                for axis_value in axis_times
                            ),
                            key=lambda item: item[0],
                        )
                        x, y, noise = map(list, zip(*series, strict=True))

                        p = plt.plot(x, y, shape, marker="o", label=label)

                        def plot_confidence_band(first, last):
                            if last - first < 2:
                                return

                            band_x = x[first:last]
                            band_y = y[first:last]
                            band_noise = noise[first:last]
                            top = [
                                band_y[i] + band_y[i] * band_noise[i]
                                for i in range(len(band_x))
                            ]
                            bottom = [
                                max(
                                    band_y[i] - band_y[i] * band_noise[i],
                                    band_y[i] * 0.001,
                                )
                                for i in range(len(band_x))
                            ]
                            plt.fill_between(
                                band_x, bottom, top, color=p[0].get_color(), alpha=0.1
                            )

                        start = None
                        for i, noise_value in enumerate(noise):
                            if has_finite_noise(noise_value) and start is None:
                                start = i
                            if not has_finite_noise(noise_value) and start is not None:
                                plot_confidence_band(start, i)
                                start = None

                        if start is not None:
                            plot_confidence_band(start, len(x))

                    for axis in plot_data["cmp"].keys():
                        plot_line("cmp", "-", axis, axis)
                        plot_line("ref", "--", axis + " ref", axis)

                    plt.legend()
                    plt.show()
                finally:
                    plt.close(fig)

    if plot:
        title = "%SOL Bandwidth change"
        if len(comparison_device_names) == 1:
            title = f"{title} - {next(iter(comparison_device_names))}"
        if filter_plan.global_axis_filters:
            axis_label = ", ".join(
                axis_filter["display"]
                for axis_filter in filter_plan.global_axis_filters
                if len(axis_filter["values"]) == 1
            )
            if axis_label:
                title = f"{title} ({axis_label})"
        plot_comparison_entries(comparison_entries, title=title, dark=dark)


def main() -> int:
    """
    Returns a process exit code.
      - 0 means the comparison completed successfully.
      - 1 signals an error has occurred.

    The number of detected regressions is reported in the summary output.
    """
    help_text = "%(prog)s [reference.json compare.json | reference_dir/ compare_dir/]"
    parser = argparse.ArgumentParser(prog="nvbench_compare", usage=help_text)
    parser.add_argument(
        "--ignore-devices",
        dest="ignore_devices",
        default=False,
        help="Ignore differences in the device sections and compare anyway",
        action="store_true",
    )
    parser.add_argument(
        "--threshold-diff",
        type=float,
        dest="threshold",
        default=0.0,
        help="only show benchmarks where percentage diff is >= THRESHOLD",
    )
    parser.add_argument(
        "--plot-along", type=str, dest="plot_along", default=None, help="plot results"
    )
    parser.add_argument(
        "--plot",
        dest="plot",
        default=False,
        help="plot comparison summary",
        action="store_true",
    )
    parser.add_argument(
        "--dark",
        action="store_true",
        help="Use dark theme (black background, white text)",
    )
    parser.add_argument(
        "--no-color",
        dest="no_color",
        action="store_true",
        help="Use emoji instead of ANSI color codes (useful for GitHub issues/PRs)",
    )
    parser.add_argument(
        "--reference-devices",
        default="all",
        help="Reference devices to compare: all, a non-negative integer id, or comma-separated ids",
    )
    parser.add_argument(
        "--compare-devices",
        default="all",
        help="Compare devices to compare: all, a non-negative integer id, or comma-separated ids",
    )
    parser.add_argument(
        "-a",
        "--axis",
        dest="filter_actions",
        action=OrderedBenchmarkFilterAction,
        help=(
            "Filter on axis value, e.g. -a Elements{io}=2^20. Applies to the "
            "most recent --benchmark, or all benchmarks if specified before any "
            "--benchmark arguments."
        ),
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        dest="filter_actions",
        action=OrderedBenchmarkFilterAction,
        help="Filter by benchmark name (can repeat)",
    )

    args, files_or_dirs = parser.parse_known_args()
    try:
        filter_plan = build_benchmark_filter_plan(args.filter_actions)
        reference_device_filter = parse_device_filter(
            args.reference_devices, "--reference-devices"
        )
        compare_device_filter = parse_device_filter(
            args.compare_devices, "--compare-devices"
        )
    except ValueError as exc:
        print(str(exc))
        return 1

    if len(files_or_dirs) != 2:
        parser.print_help()
        return 1

    # if provided two directories, find all the exactly named files
    # in both and treat them as the reference and compare
    to_compare = []
    if os.path.isdir(files_or_dirs[0]) and os.path.isdir(files_or_dirs[1]):
        for f in os.listdir(files_or_dirs[1]):
            if os.path.splitext(f)[1] != ".json":
                continue
            r = os.path.join(files_or_dirs[0], f)
            c = os.path.join(files_or_dirs[1], f)
            if (
                os.path.isfile(r)
                and os.path.isfile(c)
                and os.path.getsize(r) > 0
                and os.path.getsize(c) > 0
            ):
                to_compare.append((r, c))
    else:
        to_compare = [(files_or_dirs[0], files_or_dirs[1])]

    stats = ComparisonStats()

    for ref, comp in to_compare:
        ref_root = reader.read_file(ref)
        cmp_root = reader.read_file(comp)

        try:
            selected_ref_devices = select_devices(
                ref_root["devices"], reference_device_filter, "--reference-devices"
            )
            selected_cmp_devices = select_devices(
                cmp_root["devices"], compare_device_filter, "--compare-devices"
            )
        except ValueError as exc:
            print(str(exc))
            return 1

        if len(selected_ref_devices) != len(selected_cmp_devices):
            print(
                f"--reference-devices selected {len(selected_ref_devices)} device(s), "
                f"but --compare-devices selected {len(selected_cmp_devices)} device(s)"
            )
            return 1

        if selected_ref_devices != selected_cmp_devices:
            warn_fore = Fore.YELLOW if args.ignore_devices else Fore.RED
            msg_text = "Device sections do not match"
            print(colorize(msg_text, warn_fore, Emoji.NONE, args.no_color), end="")
            print(": ", end="")

            print(
                jsondiff.diff(
                    selected_ref_devices, selected_cmp_devices, syntax="symmetric"
                )
            )
            if not args.ignore_devices and require_matching_device_sections(
                reference_device_filter, compare_device_filter
            ):
                return 1

        run_data = ComparisonRunData(
            stats=stats,
            ref_devices=tuple(selected_ref_devices),
            cmp_devices=tuple(selected_cmp_devices),
        )

        try:
            compare_benches(
                run_data,
                ref_root["benchmarks"],
                cmp_root["benchmarks"],
                args.threshold,
                args.plot_along,
                args.plot,
                args.dark,
                filter_plan,
                args.no_color,
                reference_device_filter,
                compare_device_filter,
                os.path.dirname(ref),
                os.path.dirname(comp),
            )
        except ValueError as exc:
            print(str(exc))
            return 1

    print("# Summary\n")
    print(f"- Total Matches: {stats.config_count}")
    print(f"  - Pass        (abs(%Diff) <= max_noise): {stats.pass_count}")
    print(
        "  - Improvement (abs(%Diff) > max_noise, %Diff < 0): "
        f"{stats.improvement_count}"
    )
    print(
        f"  - Regression  (abs(%Diff) > max_noise, %Diff > 0): {stats.regression_count}"
    )
    print(
        f"  - Undecided   (comparison requires more evidence): {stats.undecided_count}"
    )
    print(f"  - Unknown     (infinite or unavailable noise): {stats.unknown_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
