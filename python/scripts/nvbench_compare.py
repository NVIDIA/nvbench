#!/usr/bin/env python

import argparse
import math
import os
import sys

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
failure_count = 0
pass_count = 0


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
        return "2^%d" % value
    return "%d" % value


def format_float64_axis_value(axis_name, axis_value, axes):
    return "%.5g" % float(axis_value["value"])


def format_type_axis_value(axis_name, axis_value, axes):
    return "%s" % axis_value["value"]


def format_string_axis_value(axis_name, axis_value, axes):
    return "%s" % axis_value["value"]


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
            raise ValueError("Axis filter must be NAME=VALUE: {}".format(axis_arg))
        name, value = axis_arg.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name or not value:
            raise ValueError("Axis filter must be NAME=VALUE: {}".format(axis_arg))

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

        display = make_display(name, display_values)
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
    return "%0.3f %s" % (seconds * multiplier, units)


def format_percentage(percentage):
    # When there aren't enough samples for a meaningful noise measurement,
    # the noise is recorded as infinity. Unfortunately, JSON spec doesn't
    # allow for inf, so these get turned into null.
    if percentage is None:
        return "inf"
    return "%0.2f%%" % (percentage * 100.0)


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
        print("Saved comparison plot to {}".format(output))
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
    axis_filters,
    benchmark_filters,
):
    if plot_along:
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set()

    comparison_entries = []
    comparison_device_names = set()
    for cmp_bench in cmp_benches:
        ref_bench = find_matching_bench(cmp_bench, ref_benches)
        if not ref_bench:
            continue
        if benchmark_filters and cmp_bench["name"] not in benchmark_filters:
            continue

        print("# %s\n" % (cmp_bench["name"]))

        cmp_device_ids = cmp_bench["devices"]
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

        for cmp_device_id in cmp_device_ids:
            rows = []
            plot_data = {"cmp": {}, "ref": {}, "cmp_noise": {}, "ref_noise": {}}

            for cmp_state in cmp_states:
                cmp_state_name = cmp_state["name"]
                ref_state = next(
                    filter(lambda st: st["name"] == cmp_state_name, ref_states), None
                )
                if not ref_state:
                    continue
                if not matches_axis_filters(cmp_state, axis_filters):
                    continue

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

                def lookup_summary(summaries, tag):
                    return next(filter(lambda s: s["tag"] == tag, summaries), None)

                cmp_time_summary = lookup_summary(
                    cmp_summaries, "nv/cold/time/gpu/mean"
                )
                ref_time_summary = lookup_summary(
                    ref_summaries, "nv/cold/time/gpu/mean"
                )
                cmp_noise_summary = lookup_summary(
                    cmp_summaries, "nv/cold/time/gpu/stdev/relative"
                )
                ref_noise_summary = lookup_summary(
                    ref_summaries, "nv/cold/time/gpu/stdev/relative"
                )

                # TODO: Use other timings, too. Maybe multiple rows, with a
                # "Timing" column + values "CPU/GPU/Batch"?
                if not all(
                    [
                        cmp_time_summary,
                        ref_time_summary,
                        cmp_noise_summary,
                        ref_noise_summary,
                    ]
                ):
                    continue

                def extract_value(summary):
                    summary_data = summary["data"]
                    value_data = next(
                        filter(lambda v: v["name"] == "value", summary_data)
                    )
                    assert value_data["type"] == "float64"
                    return value_data["value"]

                cmp_time = extract_value(cmp_time_summary)
                ref_time = extract_value(ref_time_summary)
                cmp_noise = extract_value(cmp_noise_summary)
                ref_noise = extract_value(ref_noise_summary)

                # Convert string encoding to expected numerics:
                cmp_time = float(cmp_time)
                ref_time = float(ref_time)

                diff = cmp_time - ref_time
                frac_diff = diff / ref_time

                if ref_noise and cmp_noise:
                    ref_noise = float(ref_noise)
                    cmp_noise = float(cmp_noise)
                    min_noise = min(ref_noise, cmp_noise)
                elif ref_noise:
                    ref_noise = float(ref_noise)
                    min_noise = ref_noise
                elif cmp_noise:
                    cmp_noise = float(cmp_noise)
                    min_noise = cmp_noise
                else:
                    min_noise = None  # Noise is inf

                if plot_along:
                    axis_name = []
                    axis_value = "--"
                    for aid in range(len(axis_values)):
                        if axis_values[aid]["name"] != plot_along:
                            axis_name.append(
                                "{} = {}".format(
                                    axis_values[aid]["name"], axis_values[aid]["value"]
                                )
                            )
                        else:
                            axis_value = float(axis_values[aid]["value"])
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
                global failure_count

                config_count += 1
                if not min_noise:
                    unknown_count += 1
                    status_label = "????"
                    status = Fore.YELLOW + status_label + Fore.RESET
                elif abs(frac_diff) <= min_noise:
                    pass_count += 1
                    status_label = "SAME"
                    status = Fore.BLUE + status_label + Fore.RESET
                elif diff < 0:
                    failure_count += 1
                    status_label = "FAST"
                    status = Fore.GREEN + status_label + Fore.RESET
                else:
                    failure_count += 1
                    status_label = "SLOW"
                    status = Fore.RED + status_label + Fore.RESET

                if abs(frac_diff) >= threshold:
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
                            label = "{} | {}".format(cmp_bench["name"], axis_label)
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
            ref_device = find_device_by_id(ref_state["device"], all_ref_devices)

            if cmp_device == ref_device:
                print("## [%d] %s\n" % (cmp_device["id"], cmp_device["name"]))
            else:
                print(
                    "## [%d] %s vs. [%d] %s\n"
                    % (
                        ref_device["id"],
                        ref_device["name"],
                        cmp_device["id"],
                        cmp_device["name"],
                    )
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

                def plot_line(key, shape, label):
                    x = [float(x) for x in plot_data[key][axis].keys()]
                    y = list(plot_data[key][axis].values())

                    noise = list(plot_data[key + "_noise"][axis].values())

                    top = [y[i] + y[i] * noise[i] for i in range(len(x))]
                    bottom = [y[i] - y[i] * noise[i] for i in range(len(x))]

                    p = plt.plot(x, y, shape, marker="o", label=label)
                    plt.fill_between(x, bottom, top, color=p[0].get_color(), alpha=0.1)

                for axis in plot_data["cmp"].keys():
                    plot_line("cmp", "-", axis)
                    plot_line("ref", "--", axis + " ref")

                plt.legend()
                plt.show()

    if plot:
        title = "%SOL Bandwidth change"
        if len(comparison_device_names) == 1:
            title = "{} - {}".format(title, next(iter(comparison_device_names)))
        if axis_filters:
            axis_label = ", ".join(
                axis_filter["display"]
                for axis_filter in axis_filters
                if len(axis_filter["values"]) == 1
            )
            if axis_label:
                title = "{} ({})".format(title, axis_label)
        plot_comparison_entries(comparison_entries, title=title, dark=dark)


def main():
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
        "-a",
        "--axis",
        action="append",
        default=[],
        help="Filter on axis value, e.g. -a Elements{io}=2^20 (can repeat)",
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        action="append",
        default=[],
        help="Filter by benchmark name (can repeat)",
    )

    args, files_or_dirs = parser.parse_known_args()
    print(files_or_dirs)
    try:
        axis_filters = parse_axis_filters(args.axis)
    except ValueError as exc:
        print(str(exc))
        sys.exit(1)

    if len(files_or_dirs) != 2:
        parser.print_help()
        sys.exit(1)

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
        all_ref_devices = ref_root["devices"]
        all_cmp_devices = cmp_root["devices"]

        if ref_root["devices"] != cmp_root["devices"]:
            print(
                (Fore.YELLOW if args.ignore_devices else Fore.RED)
                + "Device sections do not match:"
                + Fore.RESET
            )
            print(
                jsondiff.diff(
                    ref_root["devices"], cmp_root["devices"], syntax="symmetric"
                )
            )
            if not args.ignore_devices:
                sys.exit(1)

        compare_benches(
            ref_root["benchmarks"],
            cmp_root["benchmarks"],
            args.threshold,
            args.plot_along,
            args.plot,
            args.dark,
            axis_filters,
            args.benchmark,
        )

    print("# Summary\n")
    print("- Total Matches: %d" % config_count)
    print("  - Pass    (diff <= min_noise): %d" % pass_count)
    print("  - Unknown (infinite noise):    %d" % unknown_count)
    print("  - Failure (diff > min_noise):  %d" % failure_count)
    return failure_count


if __name__ == "__main__":
    sys.exit(main())
