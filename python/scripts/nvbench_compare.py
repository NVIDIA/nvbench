#!/usr/bin/env python
#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from itertools import islice

import jsondiff
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

all_ref_devices = []
all_cmp_devices = []
config_count = 0
unknown_count = 0
improvement_count = 0
regression_count = 0
pass_count = 0

GPU_TIME_MIN_TAG = "nv/cold/time/gpu/min"
GPU_TIME_MAX_TAG = "nv/cold/time/gpu/max"
GPU_TIME_MEAN_TAG = "nv/cold/time/gpu/mean"
GPU_TIME_STDEV_TAG = "nv/cold/time/gpu/stdev/absolute"
GPU_TIME_STDEV_RELATIVE_TAG = "nv/cold/time/gpu/stdev/relative"
GPU_TIME_MEDIAN_TAG = "nv/cold/time/gpu/median"
GPU_TIME_IR_TAG = "nv/cold/time/gpu/ir/absolute"
GPU_TIME_IR_RELATIVE_TAG = "nv/cold/time/gpu/ir/relative"


@dataclass(frozen=True)
class GpuTimeSummary:
    minimum: float | None
    maximum: float | None
    mean: float | None
    stdev: float | None
    stdev_relative: float | None
    median: float | None
    interquartile_range: float | None
    interquartile_range_relative: float | None


@dataclass(frozen=True)
class TimeEstimate:
    center: float | None
    relative_dispersion: float | None


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


def state_name_counts(states):
    return Counter(state_match_key(state) for state in states)


def format_device_ids(device_ids):
    return ", ".join(str(device_id) for device_id in device_ids)


def parse_device_filter(device_arg, option_name):
    device_arg = device_arg.strip()
    if device_arg.lower() == "all":
        return None

    values = [value.strip() for value in device_arg.split(",")]
    if not all(values):
        raise ValueError(
            f"{option_name} must be 'all', an integer, or comma-separated integers"
        )

    try:
        return [int(value) for value in values]
    except ValueError as exc:
        raise ValueError(
            f"{option_name} must be 'all', an integer, or comma-separated integers"
        ) from exc


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
        return bench["devices"]

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


def extract_summary_value(summary):
    summary_tag = summary.get("tag", "<unknown>")
    for value_data in summary.get("data", []):
        if value_data.get("name") != "value":
            continue

        value_type = value_data.get("type")
        if value_type != "float64":
            raise ValueError(
                f"summary {summary_tag!r} field 'value' has type "
                f"{value_type!r}; expected 'float64'"
            )
        if "value" not in value_data:
            raise ValueError(f"summary {summary_tag!r} field 'value' is missing value")
        return value_data["value"]

    raise ValueError(f"summary {summary_tag!r} is missing field 'value'")


def normalize_float_value(value, *, null_value=None):
    if value is None:
        return null_value
    return float(value)


def extract_summary_float(summaries, tag, *, null_value=None):
    summary = lookup_summary(summaries, tag)
    if summary is None:
        return None
    return normalize_float_value(extract_summary_value(summary), null_value=null_value)


def extract_gpu_time_summary(summaries):
    return GpuTimeSummary(
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


def compute_common_time_estimates(ref_summary, cmp_summary):
    if has_robust_estimate(ref_summary) and has_robust_estimate(cmp_summary):
        return (
            TimeEstimate(
                center=ref_summary.median,
                relative_dispersion=select_relative_dispersion(
                    ref_summary.interquartile_range_relative,
                    ref_summary.interquartile_range,
                    ref_summary.median,
                ),
            ),
            TimeEstimate(
                center=cmp_summary.median,
                relative_dispersion=select_relative_dispersion(
                    cmp_summary.interquartile_range_relative,
                    cmp_summary.interquartile_range,
                    cmp_summary.median,
                ),
            ),
        )

    if has_mean_estimate(ref_summary) and has_mean_estimate(cmp_summary):
        return (
            TimeEstimate(
                center=ref_summary.mean,
                relative_dispersion=select_relative_dispersion(
                    ref_summary.stdev_relative, ref_summary.stdev, ref_summary.mean
                ),
            ),
            TimeEstimate(
                center=cmp_summary.mean,
                relative_dispersion=select_relative_dispersion(
                    cmp_summary.stdev_relative, cmp_summary.stdev, cmp_summary.mean
                ),
            ),
        )

    return (
        TimeEstimate(
            center=ref_summary.mean,
            relative_dispersion=compute_relative_dispersion(
                ref_summary.stdev, ref_summary.mean
            ),
        ),
        TimeEstimate(
            center=cmp_summary.mean,
            relative_dispersion=compute_relative_dispersion(
                cmp_summary.stdev, cmp_summary.mean
            ),
        ),
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


def make_display(name: str, display_values: [list[str]]) -> str:
    open_bracket, close_bracket = ("[", "]") if len(display_values) > 1 else ("", "")
    display_values = ",".join(display_values)
    return f"{name}={open_bracket}{display_values}{close_bracket}"


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
            ref_state_counts = state_name_counts(ref_device_states)
            cmp_state_counts = state_name_counts(cmp_device_states)
            if ref_state_counts != cmp_state_counts:
                raise ValueError(
                    f"benchmark {cmp_bench['name']!r} device pair "
                    f"ref={ref_device_id} cmp={cmp_device_id} has mismatched "
                    f"state occurrences: ref={dict(ref_state_counts)}, "
                    f"cmp={dict(cmp_state_counts)}"
                )

            rows = []
            plot_data = {"cmp": {}, "ref": {}, "cmp_noise": {}, "ref_noise": {}}
            counters = {}

            for cmp_state in cmp_device_states:
                cmp_state_name = state_match_key(cmp_state)
                counters[cmp_state_name] = counters.get(cmp_state_name, 0) + 1
                # Duplicate state names are matched by occurrence order within
                # the same device section.
                ref_state = next(
                    islice(
                        (
                            st
                            for st in ref_device_states
                            if state_match_key(st) == cmp_state_name
                        ),
                        counters[cmp_state_name] - 1,
                        None,
                    ),
                    None,
                )
                assert ref_state is not None, (
                    "invariant: count validation ensures match exists"
                )
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
                cmp_gpu_time = extract_gpu_time_summary(cmp_summaries)
                ref_gpu_time = extract_gpu_time_summary(ref_summaries)
                ref_estimate, cmp_estimate = compute_common_time_estimates(
                    ref_gpu_time, cmp_gpu_time
                )

                cmp_time = cmp_estimate.center
                ref_time = ref_estimate.center

                if cmp_time is None or ref_time is None:
                    continue

                if cmp_time <= 0.0 or ref_time <= 0.0:
                    continue

                cmp_noise = cmp_estimate.relative_dispersion
                ref_noise = ref_estimate.relative_dispersion

                diff = cmp_time - ref_time
                frac_diff = diff / ref_time

                if not has_finite_noise(ref_noise) or not has_finite_noise(cmp_noise):
                    max_noise = None
                else:
                    max_noise = max(ref_noise, cmp_noise)

                if plot_along:
                    axis_name = []
                    axis_value = "--"
                    for av in axis_values:
                        if av["name"] != plot_along:
                            axis_name.append(f"""{av["name"]} = {av["value"]}""")
                        else:
                            axis_value = float(av["value"])
                    axis_name = ", ".join(axis_name)

                    if axis_name not in plot_data["cmp"]:
                        plot_data["cmp"][axis_name] = {}
                        plot_data["ref"][axis_name] = {}
                        plot_data["cmp_noise"][axis_name] = {}
                        plot_data["ref_noise"][axis_name] = {}

                    plot_data["cmp"][axis_name][axis_value] = cmp_time
                    plot_data["ref"][axis_name][axis_value] = ref_time
                    plot_data["cmp_noise"][axis_name][axis_value] = cmp_noise
                    plot_data["ref_noise"][axis_name][axis_value] = ref_noise

                global config_count
                global unknown_count
                global pass_count
                global improvement_count
                global regression_count

                config_count += 1
                if max_noise is None:
                    unknown_count += 1
                    status_label = "????"
                    status = colorize(status_label, Fore.YELLOW, Emoji.YELLOW, no_color)
                elif abs(frac_diff) <= max_noise:
                    pass_count += 1
                    status_label = "SAME"
                    status = colorize(status_label, Fore.BLUE, Emoji.BLUE, no_color)
                elif diff < 0:
                    improvement_count += 1
                    status_label = "FAST"
                    status = colorize(status_label, Fore.GREEN, Emoji.GREEN, no_color)
                else:
                    regression_count += 1
                    status_label = "SLOW"
                    status = colorize(status_label, Fore.RED, Emoji.RED, no_color)

                if abs(frac_diff) >= threshold:
                    axis_filters = matching_axis_filters(cmp_state, axis_filter_groups)
                    row.append(format_duration(ref_time))
                    row.append(format_percentage(ref_noise))
                    row.append(format_duration(cmp_time))
                    row.append(format_percentage(cmp_noise))
                    row.append(format_duration(diff))
                    row.append(format_percentage(frac_diff))
                    row.append(status)

                    rows.append(row)
                    if plot:
                        axis_label = format_axis_values(axis_values, axes, axis_filters)
                        if axis_label:
                            label = f"""{cmp_bench["name"]} | {axis_label}"""
                        else:
                            label = cmp_bench["name"]
                        cmp_device = find_device_by_id(
                            cmp_state["device"], all_cmp_devices
                        )
                        if cmp_device:
                            comparison_device_names.add(cmp_device["name"])
                        comparison_entries.append(
                            (label, frac_diff, status_label, cmp_bench["name"])
                        )

            if len(rows) == 0:
                continue

            cmp_device = find_device_by_id(cmp_device_id, all_cmp_devices)
            ref_device = find_device_by_id(ref_device_id, all_ref_devices)

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
                            band_y[i] - band_y[i] * band_noise[i]
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
    Returns regression_count.
      - 0 means no slow-downs detected.
      - Positive return value corresponds to the number of slow-downs detected.
      - -1 signals an error has occurred.
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
        help="Reference devices to compare: all, an integer id, or comma-separated ids",
    )
    parser.add_argument(
        "--compare-devices",
        default="all",
        help="Compare devices to compare: all, an integer id, or comma-separated ids",
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
        return -1

    if len(files_or_dirs) != 2:
        parser.print_help()
        return -1

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

    for ref, comp in to_compare:
        ref_root = reader.read_file(ref)
        cmp_root = reader.read_file(comp)

        global all_ref_devices
        global all_cmp_devices
        try:
            all_ref_devices = select_devices(
                ref_root["devices"], reference_device_filter, "--reference-devices"
            )
            all_cmp_devices = select_devices(
                cmp_root["devices"], compare_device_filter, "--compare-devices"
            )
        except ValueError as exc:
            print(str(exc))
            return -1

        if len(all_ref_devices) != len(all_cmp_devices):
            print(
                f"--reference-devices selected {len(all_ref_devices)} device(s), "
                f"but --compare-devices selected {len(all_cmp_devices)} device(s)"
            )
            return -1

        if all_ref_devices != all_cmp_devices:
            warn_fore = Fore.YELLOW if args.ignore_devices else Fore.RED
            msg_text = "Device sections do not match"
            print(colorize(msg_text, warn_fore, Emoji.NONE, args.no_color), end="")
            print(": ", end="")

            print(jsondiff.diff(all_ref_devices, all_cmp_devices, syntax="symmetric"))
            if not args.ignore_devices and require_matching_device_sections(
                reference_device_filter, compare_device_filter
            ):
                return -1

        try:
            compare_benches(
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
            )
        except ValueError as exc:
            print(str(exc))
            return -1

    print("# Summary\n")
    print(f"- Total Matches: {config_count}")
    print(f"  - Pass        (abs(%Diff) <= max_noise): {pass_count}")
    print(f"  - Improvement (abs(%Diff) > max_noise, %Diff < 0): {improvement_count}")
    print(f"  - Regression  (abs(%Diff) > max_noise, %Diff > 0): {regression_count}")
    print(f"  - Unknown     (infinite or unavailable noise): {unknown_count}")
    return regression_count


if __name__ == "__main__":
    sys.exit(main())
