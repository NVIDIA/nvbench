#!/usr/bin/env python

import argparse
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

try:
    from nvbench_json import reader
except ImportError:
    from scripts.nvbench_json import reader

UTILIZATION_TAG = "nv/cold/bw/global/utilization"


def parse_files():
    help_text = "%(prog)s [nvbench.out.json | dir/] ..."
    parser = argparse.ArgumentParser(prog="nvbench_plot", usage=help_text)
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Save plot to this file instead of showing it",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional plot title",
    )
    parser.add_argument(
        "--dark",
        action="store_true",
        help="Use dark theme (black background, white text)",
    )
    parser.add_argument(
        "-a",
        "--axis",
        action="append",
        default=[],
        help="Filter on axis value, e.g. -a T{ct}=I8 (can repeat)",
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        action="append",
        default=[],
        help="Filter by benchmark name (can repeat)",
    )
    args, files_or_dirs = parser.parse_known_args()

    filenames = []
    for file_or_dir in files_or_dirs:
        if os.path.isdir(file_or_dir):
            for f in os.listdir(file_or_dir):
                filename = os.path.join(file_or_dir, f)
                if os.path.isfile(filename) and os.path.getsize(filename) > 0:
                    filenames.append(filename)
        else:
            assert os.path.isfile(file_or_dir)
            filenames.append(file_or_dir)

    filenames.sort()

    if not filenames:
        parser.print_help()
        sys.exit(0)

    return args, filenames


def extract_utilization(state):
    summaries = state.get("summaries") or []
    summary = next(
        filter(lambda s: s["tag"] == UTILIZATION_TAG, summaries),
        None,
    )
    if not summary:
        return None

    value_data = next(
        filter(lambda v: v["name"] == "value", summary["data"]),
        None,
    )
    if not value_data:
        return None

    return float(value_data["value"])


def parse_axis_filters(axis_args):
    filters = []
    for axis_arg in axis_args:
        if "=" not in axis_arg:
            raise ValueError("Axis filter must be NAME=VALUE: {}".format(axis_arg))
        name, value = axis_arg.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name or not value:
            raise ValueError("Axis filter must be NAME=VALUE: {}".format(axis_arg))

        values = []
        display_values = []
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            if inner:
                values = [
                    stripped for item in inner.split(",") if (stripped := item.strip())
                ]
            else:
                values = []
        else:
            values = [value]
        display_values = list(values)

        if name.endswith("[pow2]"):
            name = name[: -len("[pow2]")].strip()
            if not name:
                raise ValueError(
                    "Axis filter missing name before [pow2]: {}".format(axis_arg)
                )
            try:
                exponents = [int(v) for v in values]
            except ValueError as exc:
                raise ValueError(
                    "Axis filter [pow2] value must be integer: {}".format(axis_arg)
                ) from exc
            values = [str(2**exponent) for exponent in exponents]
            display_values = ["2^{}".format(exponent) for exponent in exponents]

        if not values:
            raise ValueError(
                "Axis filter must specify at least one value: {}".format(axis_arg)
            )

        if len(display_values) == 1:
            display = "{}={}".format(name, display_values[0])
        else:
            display = "{}=[{}]".format(name, ",".join(display_values))
        filters.append(
            {
                "name": name,
                "values": values,
                "display": display,
            }
        )
    return filters


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


def strip_axis_filters_from_state_name(state_name, axis_filters):
    if not axis_filters:
        return state_name

    tokens = state_name.split()
    filter_prefixes = set(
        "{}=".format(axis_filter["name"])
        for axis_filter in axis_filters
        if len(axis_filter["values"]) == 1
    )
    tokens = [
        token
        for token in tokens
        if not any(token.startswith(prefix) for prefix in filter_prefixes)
    ]
    return " ".join(tokens)


def collect_entries(
    filename: str, axis_filters: list[dict], benchmark_filters: list[str]
) -> tuple[list[tuple[str, float, str]], set[str]]:
    json_root = reader.read_file(filename)
    entries = []
    device_names = set()
    devices = {device["id"]: device["name"] for device in json_root.get("devices", [])}
    for bench in json_root["benchmarks"]:
        bench_name = bench["name"]
        if benchmark_filters and bench_name not in benchmark_filters:
            continue
        for state in bench["states"]:
            if not matches_axis_filters(state, axis_filters):
                continue
            utilization = extract_utilization(state)
            if utilization is None:
                continue

            state_name = state["name"]
            if state_name.startswith("Device="):
                parts = state_name.split(" ", 1)
                if len(parts) == 2:
                    state_name = parts[1]
            state_name = strip_axis_filters_from_state_name(state_name, axis_filters)
            label = "{} | {}".format(bench_name, state_name)
            device_name = devices.get(state.get("device"))
            if device_name:
                device_names.add(device_name)
            entries.append((label, utilization, bench_name))

    return entries, device_names


def plot_entries(entries, title=None, output=None, dark=False):
    if not entries:
        print("No utilization data found.")
        return 1

    labels, values, bench_names = map(list, zip(*entries))
    unique_benches = list(set(bench_names))

    cmap = plt.get_cmap("tab20", max(len(unique_benches), 1))
    bench_colors = {bench: cmap(index) for index, bench in enumerate(unique_benches)}
    colors = [bench_colors[bench] for bench in bench_names]

    fig_height = max(4.0, 0.3 * len(entries) + 1.5)
    style = "dark_background" if dark else "default"
    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=(10, fig_height))

        y_pos = range(len(labels))
        ax.barh(y_pos, values, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_ylim(len(labels) - 0.5, -0.5)

        ax.xaxis.set_major_formatter(PercentFormatter(1.0))

        if title:
            ax.set_title(title)

        fig.tight_layout()

        if output:
            fig.savefig(output, dpi=150)
        else:
            plt.show()

    return 0


def main():
    args, filenames = parse_files()
    try:
        axis_filters = parse_axis_filters(args.axis)
    except ValueError as exc:
        print(str(exc))
        return 1
    entries = []
    device_names = set()
    for filename in filenames:
        file_entries, file_device_names = collect_entries(
            filename,
            axis_filters,
            args.benchmark,
        )
        entries.extend(file_entries)
        device_names.update(file_device_names)

    title = args.title
    if title is None:
        title = "%SOL Bandwidth"
        if len(device_names) == 1:
            title = "{} - {}".format(title, next(iter(device_names)))
    if axis_filters:
        axis_label = ", ".join(axis_filter["display"] for axis_filter in axis_filters)
        title = "{} ({})".format(title, axis_label)

    return plot_entries(entries, title=title, output=args.output, dark=args.dark)


if __name__ == "__main__":
    sys.exit(main())
