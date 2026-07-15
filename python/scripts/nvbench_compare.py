#!/usr/bin/env python
#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

if __package__:
    from .nvbench_json import reader
    from .nvbench_tooling_deps import (
        MissingToolingDependencyError,
        ToolingDependency,
        require_tooling_dependency,
    )
else:
    from nvbench_json import reader  # type: ignore[no-redef]
    from nvbench_tooling_deps import (  # type: ignore[no-redef]
        MissingToolingDependencyError,
        ToolingDependency,
        require_tooling_dependency,
    )


# Parse version string into tuple, "x.y.z" -> (x, y, z)
def version_tuple(v):
    return tuple(map(int, (v.split("."))))


Fore: Any = None


def load_nvbench_compare_tooling(*, load_color: bool = True) -> None:
    global Fore

    if load_color and Fore is None:
        colorama = require_tooling_dependency(
            ToolingDependency(
                "colorama", "colorama", "colored status output", extra="compare"
            ),
            tool_name="nvbench-compare-legacy",
        )
        Fore = colorama.Fore


def load_tabulate_for_table_output() -> tuple[Any, tuple[int, ...]]:
    tabulate_module = require_tooling_dependency(
        ToolingDependency("tabulate", "tabulate", "table output", extra="compare"),
        tool_name="nvbench-compare-legacy",
    )
    return tabulate_module, version_tuple(tabulate_module.__version__)


def load_jsondiff_for_device_diff() -> Any:
    return require_tooling_dependency(
        ToolingDependency(
            "jsondiff", "jsondiff", "device metadata diffs", extra="compare"
        ),
        tool_name="nvbench-compare-legacy",
    )


GPU_TIME_MIN_TAG = "nv/cold/time/gpu/min"
GPU_TIME_MAX_TAG = "nv/cold/time/gpu/max"
GPU_TIME_MEAN_TAG = "nv/cold/time/gpu/mean"
GPU_TIME_STDEV_TAG = "nv/cold/time/gpu/stdev/absolute"
GPU_TIME_STDEV_RELATIVE_TAG = "nv/cold/time/gpu/stdev/relative"


def read_nvbench_json_root(filename: str) -> Mapping[str, Any]:
    try:
        root = reader.read_file(filename)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ValueError(
            f"failed to read NVBench JSON file {filename!r}: {exc}"
        ) from exc

    if not isinstance(root, Mapping):
        raise ValueError(f"NVBench JSON file {filename!r} root must be an object")

    missing_keys = [key for key in ("devices", "benchmarks") if key not in root]
    if missing_keys:
        missing = ", ".join(repr(key) for key in missing_keys)
        raise ValueError(
            f"NVBench JSON file {filename!r} is missing required root key(s): {missing}"
        )

    for key in ("devices", "benchmarks"):
        value = root[key]
        if not isinstance(value, list):
            raise ValueError(
                f"NVBench JSON file {filename!r} root key {key!r} must be an array"
            )
        for index, entry in enumerate(value):
            if not isinstance(entry, Mapping):
                raise ValueError(
                    f"NVBench JSON file {filename!r} root key {key!r} entry "
                    f"{index} must be an object"
                )

    return root


def format_json_structure_error(ref: str, comp: str, exc: Exception) -> str:
    if isinstance(exc, KeyError) and exc.args:
        detail = f"missing key {exc.args[0]!r}"
    else:
        detail = str(exc) or exc.__class__.__name__
    return (
        f"invalid NVBench JSON structure while comparing {ref!r} and {comp!r}: {detail}"
    )


@dataclass(frozen=True)
class GpuTimingData:
    mean: float | None
    stdev: float | None
    stdev_relative: float | None


@dataclass(frozen=True)
class TimeEstimate:
    center: float | None
    relative_dispersion: float | None


@dataclass(frozen=True)
class TimingInterval:
    lower: float
    upper: float
    center: float


@dataclass(frozen=True)
class TimingComparisonInputs:
    ref_estimate: TimeEstimate
    cmp_estimate: TimeEstimate
    ref_interval: TimingInterval | None
    cmp_interval: TimingInterval | None


class ComparisonStatus(str, Enum):
    UNKNOWN = "????"
    SAME = "SAME"
    FAST = "FAST"
    SLOW = "SLOW"


@dataclass(frozen=True)
class DecisionReason:
    code: str
    message: str
    severity: float = 0.0


@dataclass(frozen=True)
class TimingDecision:
    status: ComparisonStatus
    reason: DecisionReason


@dataclass(frozen=True)
class SummaryComparison:
    ref_interval: TimingInterval | None
    cmp_interval: TimingInterval | None
    ref_estimate: TimeEstimate
    cmp_estimate: TimeEstimate
    ref_time: float | None
    cmp_time: float | None
    ref_noise: float | None
    cmp_noise: float | None
    diff: float | None
    frac_diff: float | None
    diff_interval: tuple[float, float] | None
    frac_diff_interval: tuple[float, float] | None
    min_noise: float | None
    status: ComparisonStatus
    reason: DecisionReason


@dataclass
class ComparisonStats:
    config_count: int = 0
    pass_count: int = 0
    improvement_count: int = 0
    regression_count: int = 0
    unknown_count: int = 0

    def record(
        self, status: ComparisonStatus, reason: DecisionReason | None = None
    ) -> None:
        self.config_count += 1
        if status == ComparisonStatus.UNKNOWN:
            self.unknown_count += 1
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


def normalized_axis_values(state):
    axis_values = state.get("axis_values") or []
    return tuple(
        sorted(
            (
                axis_value.get("name"),
                axis_value.get("type"),
                repr(axis_value.get("value")),
            )
            for axis_value in axis_values
        )
    )


def state_comparison_key(state):
    return state_match_key(state), normalized_axis_values(state)


def group_states_by_match_key(states):
    grouped = {}
    for state in states:
        grouped.setdefault(state_comparison_key(state), []).append(state)
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
    SHRUG = "\U0001f937"
    NONE = ""


def colorize(msg: str, fore: str, emoji: Emoji, no_color: bool) -> str:
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
    if isinstance(value, bool):
        return null_value
    return float(value)


def extract_summary_float(summaries, tag, *, null_value=None):
    summary = lookup_summary(summaries, tag)
    if summary is None:
        return None
    return normalize_float_value(extract_summary_value(summary), null_value=null_value)


def extract_gpu_timing_data(summaries):
    mean = extract_summary_float(summaries, GPU_TIME_MEAN_TAG)
    stdev = extract_summary_float(summaries, GPU_TIME_STDEV_TAG, null_value=math.inf)
    stdev_relative = extract_summary_float(
        summaries, GPU_TIME_STDEV_RELATIVE_TAG, null_value=math.inf
    )
    if stdev is None:
        stdev = derive_absolute_dispersion(stdev_relative, mean)

    return GpuTimingData(
        mean=mean,
        stdev=stdev,
        stdev_relative=stdev_relative,
    )


def make_empty_gpu_timing_data():
    return GpuTimingData(
        mean=None,
        stdev=None,
        stdev_relative=None,
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


def is_finite(value):
    return value is not None and math.isfinite(value)


def is_positive_finite(value):
    return is_finite(value) and value > 0.0


def is_nonnegative_finite(value):
    return is_finite(value) and value >= 0.0


def derive_absolute_dispersion(relative_dispersion, center):
    if is_nonnegative_finite(relative_dispersion) and is_positive_finite(center):
        return relative_dispersion * center
    return None


def parse_plot_axis_value(axis_name, axis_value):
    try:
        value = float(axis_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"--plot-along requires numeric axis values; "
            f"axis {axis_name!r} has value {axis_value!r}"
        ) from exc
    if not is_positive_finite(value):
        raise ValueError(
            f"--plot-along requires positive finite axis values; "
            f"axis {axis_name!r} has value {axis_value!r}"
        )
    return value


def make_decision(status, code, message, *, severity=0.0):
    return TimingDecision(
        status=status,
        reason=DecisionReason(code=code, message=message, severity=severity),
    )


def select_relative_dispersion(relative_dispersion, absolute_dispersion, center):
    if relative_dispersion is not None:
        return relative_dispersion
    return compute_relative_dispersion(absolute_dispersion, center)


def unusable_timing_center_decision(ref_time, cmp_time):
    if ref_time is None or cmp_time is None:
        return make_decision(
            ComparisonStatus.UNKNOWN,
            "timing_center_missing",
            "timing center is missing",
        )
    if not math.isfinite(ref_time) or not math.isfinite(cmp_time):
        return make_decision(
            ComparisonStatus.UNKNOWN,
            "timing_center_nonfinite",
            "timing center is non-finite",
        )
    if ref_time <= 0.0 or cmp_time <= 0.0:
        return make_decision(
            ComparisonStatus.UNKNOWN,
            "timing_center_nonpositive",
            "timing center is non-positive",
        )
    return None


def make_unavailable_timing_comparison(decision, timing_inputs):
    return SummaryComparison(
        ref_interval=timing_inputs.ref_interval,
        cmp_interval=timing_inputs.cmp_interval,
        ref_estimate=timing_inputs.ref_estimate,
        cmp_estimate=timing_inputs.cmp_estimate,
        ref_time=timing_inputs.ref_estimate.center,
        cmp_time=timing_inputs.cmp_estimate.center,
        ref_noise=timing_inputs.ref_estimate.relative_dispersion,
        cmp_noise=timing_inputs.cmp_estimate.relative_dispersion,
        diff=None,
        frac_diff=None,
        diff_interval=None,
        frac_diff_interval=None,
        min_noise=None,
        status=decision.status,
        reason=decision.reason,
    )


def compute_legacy_timing_comparison_inputs(ref_timing, cmp_timing):
    return TimingComparisonInputs(
        ref_estimate=TimeEstimate(
            center=ref_timing.mean,
            relative_dispersion=select_relative_dispersion(
                ref_timing.stdev_relative, ref_timing.stdev, ref_timing.mean
            ),
        ),
        cmp_estimate=TimeEstimate(
            center=cmp_timing.mean,
            relative_dispersion=select_relative_dispersion(
                cmp_timing.stdev_relative, cmp_timing.stdev, cmp_timing.mean
            ),
        ),
        ref_interval=None,
        cmp_interval=None,
    )


def compare_gpu_timings(ref_timing, cmp_timing):
    timing_inputs = compute_legacy_timing_comparison_inputs(ref_timing, cmp_timing)
    ref_estimate = timing_inputs.ref_estimate
    cmp_estimate = timing_inputs.cmp_estimate

    cmp_time = cmp_estimate.center
    ref_time = ref_estimate.center

    cmp_noise = cmp_estimate.relative_dispersion
    ref_noise = ref_estimate.relative_dispersion

    unusable_center_decision = unusable_timing_center_decision(ref_time, cmp_time)
    if unusable_center_decision is not None:
        return make_unavailable_timing_comparison(
            unusable_center_decision, timing_inputs
        )

    diff = cmp_time - ref_time
    frac_diff = diff / ref_time

    if not is_usable_noise(ref_noise) or not is_usable_noise(cmp_noise):
        decision = make_decision(
            ComparisonStatus.UNKNOWN,
            "noise_unavailable",
            "relative standard deviation is unavailable, negative, or non-finite",
        )
        min_noise = None
    else:
        min_noise = min(ref_noise, cmp_noise)
        if abs(frac_diff) <= min_noise:
            decision = make_decision(
                ComparisonStatus.SAME,
                "same",
                "absolute fractional difference is within relative standard deviation",
            )
        elif diff < 0:
            decision = make_decision(
                ComparisonStatus.FAST,
                "fast",
                "compare timing mean is lower than reference timing mean",
            )
        else:
            decision = make_decision(
                ComparisonStatus.SLOW,
                "slow",
                "compare timing mean is higher than reference timing mean",
            )

    return SummaryComparison(
        ref_interval=None,
        cmp_interval=None,
        ref_estimate=ref_estimate,
        cmp_estimate=cmp_estimate,
        ref_time=ref_time,
        cmp_time=cmp_time,
        ref_noise=ref_noise,
        cmp_noise=cmp_noise,
        diff=diff,
        frac_diff=frac_diff,
        diff_interval=None,
        frac_diff_interval=None,
        min_noise=min_noise,
        status=decision.status,
        reason=decision.reason,
    )


def get_state_summaries(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    summaries = state.get("summaries")
    return summaries if summaries is not None else []


def state_has_summaries(state):
    return bool(state.get("summaries"))


def format_skipped_state_reason(side, state):
    reason = state.get("skip_reason")
    if reason:
        return f"{side} state skipped: {reason}"
    return f"{side} state skipped"


def missing_state_summaries_decision(ref_state, cmp_state):
    skipped_messages = []
    if ref_state.get("is_skipped"):
        skipped_messages.append(format_skipped_state_reason("reference", ref_state))
    if cmp_state.get("is_skipped"):
        skipped_messages.append(format_skipped_state_reason("compare", cmp_state))
    if skipped_messages:
        return make_decision(
            ComparisonStatus.UNKNOWN,
            "state_skipped",
            "; ".join(skipped_messages),
        )

    missing_sides = []
    if not state_has_summaries(ref_state):
        missing_sides.append("reference")
    if not state_has_summaries(cmp_state):
        missing_sides.append("compare")
    if not missing_sides:
        return None
    if len(missing_sides) == 2:
        message = "reference and compare GPU timing summaries are missing"
    else:
        message = f"{missing_sides[0]} GPU timing summaries are missing"
    return make_decision(
        ComparisonStatus.UNKNOWN,
        "gpu_timing_summaries_missing",
        message,
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


def find_axis_by_name(axis_name, axes):
    for axis in axes:
        if axis["name"] == axis_name:
            return axis
    raise KeyError(f"axis metadata not found for {axis_name!r}")


def format_int64_axis_value(axis_name, axis_value, axis):
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
    axis = find_axis_by_name(axis_name, axes)
    axis_type = axis["type"]
    if axis_type == "int64":
        return format_int64_axis_value(axis_name, axis_value, axis)
    elif axis_type == "float64":
        return format_float64_axis_value(axis_name, axis_value, axes)
    elif axis_type == "type":
        return format_type_axis_value(axis_name, axis_value, axes)
    elif axis_type == "string":
        return format_string_axis_value(axis_name, axis_value, axes)
    raise ValueError(f"unsupported axis type {axis_type!r} for axis {axis_name!r}")


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
    return bool(axis_filter_groups_for_benchmark(benchmark_name, filter_plan))


def axis_filter_groups_for_benchmark(benchmark_name, filter_plan):
    if not filter_plan.benchmark_scopes:
        return [filter_plan.global_axis_filters]

    matching_scopes = [
        scope
        for scope in filter_plan.benchmark_scopes
        if scope.benchmark_name == benchmark_name
    ]

    if matching_scopes:
        return [
            filter_plan.global_axis_filters + scope.axis_filters
            for scope in matching_scopes
        ]
    return []


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


def format_duration(seconds, *, allow_negative=False, allow_zero=False):
    if (
        not is_finite(seconds)
        or (seconds < 0.0 and not allow_negative)
        or (seconds == 0.0 and not allow_zero)
    ):
        return "n/a"

    magnitude = abs(seconds)
    if magnitude >= 1:
        multiplier = 1.0
        units = "s"
    elif magnitude >= 1e-3:
        multiplier = 1e3
        units = "ms"
    else:
        multiplier = 1e6
        units = "us"
    return f"{seconds * multiplier:0.3f} {units}"


def select_duration_units(*seconds_values):
    seconds_values = [value for value in seconds_values if is_finite(value)]
    if not seconds_values:
        return 1e6, "us"

    max_abs_seconds = max(abs(value) for value in seconds_values)
    if max_abs_seconds >= 1:
        return 1.0, "s"
    if max_abs_seconds >= 1e-3:
        return 1e3, "ms"
    return 1e6, "us"


def duration_precision_for_center(center, delta_multiplier):
    if not is_finite(center):
        return 3

    center_multiplier, _ = select_duration_units(center)
    center_quantum = 10.0**-3 * (delta_multiplier / center_multiplier)
    if center_quantum >= 1.0:
        return 0
    return int(math.ceil(-math.log10(center_quantum)))


def format_duration_range(bounds):
    if bounds is None:
        return "n/a"
    lower, upper = bounds
    if not is_finite(lower) or not is_finite(upper):
        return "n/a"

    multiplier, units = select_duration_units(lower, upper)
    return f"[{lower * multiplier:0.2f}, {upper * multiplier:0.2f}] {units}"


def format_percentage(percentage):
    if percentage is None:
        return "n/a"
    if math.isnan(percentage):
        return "n/a"
    if math.isinf(percentage):
        return "inf"
    return f"{percentage * 100.0:0.2f}%"


def get_display_headers():
    return (
        [
            "Ref Time",
            "Ref Noise",
            "Cmp Time",
            "Cmp Noise",
            "Diff",
            "%Diff",
            "Status",
        ],
        ["right", "right", "right", "right", "right", "right", "center"],
    )


def append_display_row(row, comparison, no_color):
    row.append(format_duration(comparison.ref_time))
    row.append(format_percentage(comparison.ref_noise))
    row.append(format_duration(comparison.cmp_time))
    row.append(format_percentage(comparison.cmp_noise))
    row.append(format_duration(comparison.diff, allow_negative=True, allow_zero=True))
    row.append(format_percentage(comparison.frac_diff))
    row.append(colorize_comparison_status(comparison.status, no_color))


def is_usable_noise(noise):
    return is_nonnegative_finite(noise)


def colorize_comparison_status(status, no_color):
    if status == ComparisonStatus.UNKNOWN:
        fore_name = "YELLOW"
        emoji = Emoji.YELLOW
    elif status == ComparisonStatus.SAME:
        fore_name = "BLUE"
        emoji = Emoji.BLUE
    elif status == ComparisonStatus.FAST:
        fore_name = "GREEN"
        emoji = Emoji.GREEN
    else:
        fore_name = "RED"
        emoji = Emoji.RED

    fore = "" if no_color else getattr(Fore, fore_name)
    return colorize(status.value, fore, emoji, no_color)


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


def format_plot_series_key(state_key, occurrence, occurrence_count, axis_name_parts):
    parts = []
    if state_key:
        parts.append(state_key)
    if occurrence_count > 1:
        parts.append(f"occurrence={occurrence + 1}/{occurrence_count}")
    parts.extend(axis_name_parts)
    return ", ".join(parts)


def plot_comparison_entries(entries, title=None, dark=False):
    if not entries:
        print("No comparison data to plot.")
        return 1

    matplotlib = require_tooling_dependency(
        ToolingDependency("matplotlib", "matplotlib", "plot rendering", extra="plot"),
        tool_name="nvbench-compare-legacy",
    )
    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")

    plt = require_tooling_dependency(
        ToolingDependency(
            "matplotlib.pyplot", "matplotlib", "plot rendering", extra="plot"
        ),
        tool_name="nvbench-compare-legacy",
    )
    ticker = require_tooling_dependency(
        ToolingDependency(
            "matplotlib.ticker", "matplotlib", "plot axis formatting", extra="plot"
        ),
        tool_name="nvbench-compare-legacy",
    )
    PercentFormatter = ticker.PercentFormatter

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
):
    if plot_along:
        plt = require_tooling_dependency(
            ToolingDependency(
                "matplotlib.pyplot",
                "matplotlib",
                "per-axis plot rendering",
                extra="plot",
            ),
            tool_name="nvbench-compare-legacy",
        )
        sns = require_tooling_dependency(
            ToolingDependency(
                "seaborn", "seaborn", "per-axis plot styling", extra="plot"
            ),
            tool_name="nvbench-compare-legacy",
        )

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
                "nvbench-compare-legacy pairs devices by position, so each compared "
                "benchmark must contain the same number of devices"
            )

        print(f"""# {cmp_bench["name"]}\n""")

        axes = cmp_bench["axes"]
        ref_states = ref_bench["states"]
        cmp_states = cmp_bench["states"]

        axes = axes if axes else []

        headers = [x["name"] for x in axes]
        colalign = ["center"] * len(headers)
        display_headers, display_colalign = get_display_headers()
        headers.extend(display_headers)
        colalign.extend(display_colalign)

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
            counters: dict[Any, int] = {}

            for cmp_state in cmp_device_states:
                cmp_state_name = state_match_key(cmp_state)
                cmp_state_key = state_comparison_key(cmp_state)
                occurrence = counters.get(cmp_state_key, 0)
                counters[cmp_state_key] = occurrence + 1
                # Duplicate state names with identical axis values are matched
                # by occurrence order within the filtered device section.
                ref_state = ref_states_by_name[cmp_state_key][occurrence]
                axis_values = cmp_state["axis_values"]
                if not axis_values:
                    axis_values = []

                row = []
                for axis_value in axis_values:
                    axis_value_name = axis_value["name"]
                    row.append(format_axis_value(axis_value_name, axis_value, axes))

                cmp_summaries = get_state_summaries(cmp_state)
                ref_summaries = get_state_summaries(ref_state)

                # TODO: Use other timings, too. Maybe multiple rows, with a
                # "Timing" column + values "CPU/GPU/Batch"?
                missing_summaries_decision = missing_state_summaries_decision(
                    ref_state, cmp_state
                )
                if missing_summaries_decision is not None:
                    ref_gpu_time = (
                        extract_gpu_timing_data(ref_summaries)
                        if ref_summaries
                        else make_empty_gpu_timing_data()
                    )
                    cmp_gpu_time = (
                        extract_gpu_timing_data(cmp_summaries)
                        if cmp_summaries
                        else make_empty_gpu_timing_data()
                    )
                    timing_inputs = compute_legacy_timing_comparison_inputs(
                        ref_gpu_time, cmp_gpu_time
                    )
                    comparison = make_unavailable_timing_comparison(
                        missing_summaries_decision, timing_inputs
                    )
                else:
                    cmp_gpu_time = extract_gpu_timing_data(cmp_summaries)
                    ref_gpu_time = extract_gpu_timing_data(ref_summaries)
                    comparison = compare_gpu_timings(ref_gpu_time, cmp_gpu_time)

                if (
                    plot_along
                    and is_positive_finite(comparison.ref_time)
                    and is_positive_finite(comparison.cmp_time)
                ):
                    axis_name_parts = []
                    axis_value = None
                    for av in axis_values:
                        if av["name"] != plot_along:
                            axis_name_parts.append(f"""{av["name"]} = {av["value"]}""")
                        else:
                            axis_value = parse_plot_axis_value(av["name"], av["value"])
                    if axis_value is not None:
                        axis_name = format_plot_series_key(
                            cmp_state_name,
                            occurrence,
                            cmp_state_counts[cmp_state_key],
                            axis_name_parts,
                        )

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

                run_data.stats.record(comparison.status, comparison.reason)
                if comparison.status == ComparisonStatus.UNKNOWN or (
                    comparison.frac_diff is not None
                    and abs(comparison.frac_diff) >= threshold
                ):
                    axis_filters = matching_axis_filters(cmp_state, axis_filter_groups)
                    append_display_row(row, comparison, no_color)

                    rows.append(row)
                    if (
                        plot
                        and comparison.frac_diff is not None
                        and math.isfinite(comparison.frac_diff)
                    ):
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
            tabulate, tabulate_version = load_tabulate_for_table_output()
            # colalign and github format require tabulate 0.8.3
            if tabulate_version >= (0, 8, 3):
                print(
                    tabulate.tabulate(
                        rows, headers=headers, colalign=colalign, tablefmt="github"
                    )
                )
            else:
                print(tabulate.tabulate(rows, headers=headers, tablefmt="pipe"))

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
                            if is_usable_noise(noise_value) and start is None:
                                start = i
                            if not is_usable_noise(noise_value) and start is not None:
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
        title = "GPU timing change"
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
    parser = argparse.ArgumentParser(usage=help_text)
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
        help="only show rows where abs(%%Diff) is >= THRESHOLD percent",
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
            "Filter on axis value, e.g. -a 'Elements{io}[pow2]=20'. Applies to the "
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
    parser.add_argument("files_or_dirs", nargs="*")

    args = parser.parse_args()
    files_or_dirs = args.files_or_dirs

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

    try:
        load_nvbench_compare_tooling(load_color=not args.no_color)
    except MissingToolingDependencyError as exc:
        print(str(exc), file=sys.stderr)
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
    if not to_compare:
        print(
            f"No non-empty matching JSON files found in {files_or_dirs[0]!r} "
            f"and {files_or_dirs[1]!r}"
        )
        return 1

    stats = ComparisonStats()

    for ref, comp in to_compare:
        try:
            ref_root = read_nvbench_json_root(ref)
            cmp_root = read_nvbench_json_root(comp)
            selected_ref_devices = select_devices(
                ref_root["devices"], reference_device_filter, "--reference-devices"
            )
            selected_cmp_devices = select_devices(
                cmp_root["devices"], compare_device_filter, "--compare-devices"
            )
        except ValueError as exc:
            print(str(exc))
            return 1
        except (AttributeError, KeyError, TypeError, IndexError) as exc:
            print(format_json_structure_error(ref, comp, exc))
            return 1

        if len(selected_ref_devices) != len(selected_cmp_devices):
            print(
                f"--reference-devices selected {len(selected_ref_devices)} device(s), "
                f"but --compare-devices selected {len(selected_cmp_devices)} device(s)"
            )
            return 1

        if selected_ref_devices != selected_cmp_devices:
            try:
                jsondiff = load_jsondiff_for_device_diff()
            except MissingToolingDependencyError as exc:
                print(str(exc), file=sys.stderr)
                return 1

            if args.no_color:
                warn_fore = ""
            else:
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
                threshold=args.threshold / 100.0,
                plot_along=args.plot_along,
                plot=args.plot,
                dark=args.dark,
                filter_plan=filter_plan,
                no_color=args.no_color,
                reference_device_filter=reference_device_filter,
                compare_device_filter=compare_device_filter,
            )
        except MissingToolingDependencyError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        except ValueError as exc:
            print(str(exc))
            return 1
        except (AttributeError, KeyError, TypeError, IndexError) as exc:
            print(format_json_structure_error(ref, comp, exc))
            return 1

    print("# Summary\n")
    print(f"- Total Matches: {stats.config_count}")
    print(f"  - Pass        (abs(%Diff) <= min_noise): {stats.pass_count}")
    print(
        "  - Improvement (abs(%Diff) > min_noise, %Diff < 0): "
        f"{stats.improvement_count}"
    )
    print(
        f"  - Regression  (abs(%Diff) > min_noise, %Diff > 0): {stats.regression_count}"
    )
    print(f"  - Unknown     (infinite or unavailable noise): {stats.unknown_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
