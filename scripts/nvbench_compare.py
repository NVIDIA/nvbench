#!/usr/bin/env python

import json
import keyword
import sys

import tabulate

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

ref_benches = ref_root["benchmarks"]
cmp_benches = cmp_root["benchmarks"]


def find_matching_bench(needle, haystack):
    for hay in haystack:
        if hay["name"] == needle["name"] and hay["axes"] == needle["axes"]:
            return hay
    return None


def find_matching_state(needle, haystack):
    for hay in haystack:
        if hay["description"] == needle["description"]:
            return hay
    return None


def find_named_value(name, named_values):
    for named_value in named_values:
        if named_value["name"] == name:
            return named_value


def get_row(cmp_benches, ref_benches):
    for cmp_bench in cmp_benches:
        ref_bench = find_matching_bench(cmp_bench, ref_benches)

        if not ref_bench:
            continue

        for cmp_state in cmp_bench["states"]:
            ref_state = find_matching_state(cmp_state, ref_bench["states"])
            if not ref_state:
                continue

            # TODO this should just be the parameterization. Refactor NVBench lib so
            # this can happen.
            state_description = cmp_state["description"]

            cmp_summaries = cmp_state["summaries"]
            ref_summaries = ref_state["summaries"]

            if not ref_summaries or not cmp_summaries:
                continue

            cmp_time_summary = cmp_summaries.get("Average GPU Time (Cold)")
            ref_time_summary = ref_summaries.get("Average GPU Time (Cold)")
            cmp_noise_summary = cmp_summaries.get("GPU Relative Standard Deviation (Cold)")
            ref_noise_summary = ref_summaries.get("GPU Relative Standard Deviation (Cold)")

            # TODO: Determine whether empty outputs could be present based on
            # user requests not to perform certain timings.
            if cmp_time_summary is None or ref_time_summary is None or \
                    cmp_noise_summary is None or ref_noise_summary is None:
                continue

            # TODO Ugly. The JSON needs to be changed to let us look up names directly.
            # Change arrays to maps.
            cmp_time = find_named_value("value", cmp_time_summary["values"])["value"]
            ref_time = find_named_value("value", ref_time_summary["values"])["value"]
            cmp_noise = find_named_value("value", cmp_noise_summary["values"])["value"]
            ref_noise = find_named_value("value", ref_noise_summary["values"])["value"]

            # Relative time comparison
            yield [cmp_bench['name'], state_description] + f"{cmp_time - ref_time} {cmp_time} {ref_time} {cmp_noise:0.6f}% {ref_noise:0.6f}%\n".split()


print(tabulate.tabulate((row for row in get_row(cmp_benches, ref_benches)),
                        floatfmt="0.12f",
                        headers=("Name", "Parameters", "Old - New", "New Time", "Old Time", "New Std", "Old Std"),
                        ))
