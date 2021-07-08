#!/usr/bin/env python

from colorama import Fore
import json
import math
import sys

import tabulate

# Parse version string into tuple, "x.y.z" -> (x, y, z)
def version_tuple(v):
    return tuple(map(int, (v.split("."))))

tabulate_version = version_tuple(tabulate.__version__)

if len(sys.argv) != 3:
    print("Usage: %s reference.json compare.json\n" % sys.argv[0])
    sys.exit(1)

with open(sys.argv[1], "r") as ref_file:
    ref_root = json.load(ref_file)

with open(sys.argv[2], "r") as cmp_file:
    cmp_root = json.load(cmp_file)

# This is blunt but works for now:
if ref_root["devices"] != cmp_root["devices"]:
    print("Device sections do not match.")
    sys.exit(1)

all_devices = cmp_root["devices"]
config_count = 0
unknown_count = 0
failure_count = 0
pass_count = 0


def find_matching_bench(needle, haystack):
    for hay in haystack:
        if hay["name"] == needle["name"] and hay["axes"] == needle["axes"]:
            return hay
    return None


def find_device_by_id(device_id):
    for device in all_devices:
        if device["id"] == device_id:
            return device
    return None


def format_int64_axis_value(axis_name, axis_value, axes):
    axis_def = axes[axis_name]
    axis_flags = axis_def["flags"]
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
    axis_def = axes[axis_name]
    axis_type = axis_def["type"]
    if axis_type == "int64":
        return format_int64_axis_value(axis_name, axis_value, axes)
    elif axis_type == "float64":
        return format_float64_axis_value(axis_name, axis_value, axes)
    elif axis_type == "type":
        return format_type_axis_value(axis_name, axis_value, axes)
    elif axis_type == "string":
        return format_string_axis_value(axis_name, axis_value, axes)


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
    if not percentage:
        return "inf"
    return "%0.2f%%" % (percentage * 100.0)


def compare_benches(ref_benches, cmp_benches):
    for cmp_bench in cmp_benches:
        ref_bench = find_matching_bench(cmp_bench, ref_benches)
        if not ref_bench:
            continue

        print("# %s\n" % (cmp_bench["name"]))

        device_ids = cmp_bench["devices"]
        axes = cmp_bench["axes"]
        ref_states = ref_bench["states"]
        cmp_states = cmp_bench["states"]

        headers = list(axes.keys()) if axes else []
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

        for device_id in device_ids:
            device = find_device_by_id(device_id)
            print("## [%d] %s\n" % (device["id"], device["name"]))

            rows = []
            for cmp_state_name in cmp_states:
                cmp_state = cmp_states[cmp_state_name]
                ref_state = ref_states[cmp_state_name]
                if not ref_state:
                    continue

                axis_values = cmp_state["axis_values"]
                if not axis_values:
                    axis_values = []

                row = []
                for axis_value_name in axis_values:
                    axis_value = axis_values[axis_value_name]
                    row.append(format_axis_value(axis_value_name,
                                                 axis_value,
                                                 axes))

                cmp_summaries = cmp_state["summaries"]
                ref_summaries = ref_state["summaries"]

                if not ref_summaries or not cmp_summaries:
                    continue

                cmp_time_summary = cmp_summaries.get("Average GPU Time (Cold)")
                ref_time_summary = ref_summaries.get("Average GPU Time (Cold)")
                cmp_noise_summary = cmp_summaries.get(
                    "GPU Relative Standard Deviation (Cold)"
                )
                ref_noise_summary = ref_summaries.get(
                    "GPU Relative Standard Deviation (Cold)"
                )

                # TODO: Use other timings, too. Maybe multiple rows, with a
                # "Timing" column + values "CPU/GPU/Batch"?
                if not all([cmp_time_summary,
                            ref_time_summary,
                            cmp_noise_summary,
                            ref_noise_summary]):
                    continue

                cmp_time = cmp_time_summary["value"]["value"]
                ref_time = ref_time_summary["value"]["value"]
                cmp_noise = cmp_noise_summary["value"]["value"]
                ref_noise = ref_noise_summary["value"]["value"]

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

                global config_count
                global unknown_count
                global pass_count
                global failure_count

                config_count += 1
                if not min_noise:
                    unknown_count += 1
                    status = Fore.YELLOW + "????" + Fore.RESET
                elif abs(frac_diff) <= min_noise:
                    pass_count += 1
                    status = Fore.GREEN + "PASS" + Fore.RESET
                else:
                    failure_count += 1
                    status = Fore.RED + "FAIL" + Fore.RESET

                row.append(format_duration(ref_time))
                row.append(format_percentage(ref_noise))
                row.append(format_duration(cmp_time))
                row.append(format_percentage(cmp_noise))
                row.append(format_duration(diff))
                row.append(format_percentage(frac_diff))
                row.append(status)

                rows.append(row)

            # colalign and github format require tabulate 0.8.3
            if tabulate_version >= (0, 8, 3):
                print(tabulate.tabulate(rows,
                                        headers=headers,
                                        colalign=colalign,
                                        tablefmt="github"))
            else:
                print(tabulate.tabulate(rows,
                                        headers=headers,
                                        tablefmt="markdown"))

            print("")


compare_benches(ref_root["benchmarks"], cmp_root["benchmarks"])

print("# Summary\n")
print("- Total Matches: %d" % config_count)
print("  - Pass    (diff <= min_noise): %d" % pass_count)
print("  - Unknown (infinite noise):    %d" % unknown_count)
print("  - Failure (diff > min_noise):  %d" % failure_count)

sys.exit(failure_count)
