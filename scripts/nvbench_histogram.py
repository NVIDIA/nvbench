#!/usr/bin/env python

import numpy as np

import argparse
import json
import os
import sys


def main():
    help_text = "%(prog)s [nvbench.out.json | dir/] ..."
    parser = argparse.ArgumentParser(prog='nvbench_histogram', usage=help_text)

    args, files_or_dirs = parser.parse_known_args()

    filenames = []
    for file_or_dir in files_or_dirs:
        if os.path.isdir(file_or_dir):
            for f in os.listdir(file_or_dir):
                if os.path.splitext(f)[1] != ".json":
                    continue
                filename = os.path.join(file_or_dir, f)
                if os.path.isfile(filename) and os.path.getsize(filename) > 0:
                    filenames.append(filename)
        else:
            filenames.append(file_or_dir)

    filenames.sort()

    for filename in filenames:
        with open(filename, "r") as f:
            json_root = json.load(f)

        for bench in json_root["benchmarks"]:
            print("Benchmark: {}".format(bench["name"]))
            for state_name, state in bench["states"].items():
                print("State: {}".format(state_name))
                try:
                    samples = state["summaries"]["nv/json/bin/nv/cold/sample_times"]
                except KeyError:
                    continue
                except TypeError:
                    continue
                sample_filename = samples["filename"]["value"]
                sample_count = int(samples["size"]["value"])

                # If not absolute, the path is relative to the associated .json file:
                if not os.path.isabs(sample_filename):
                    sample_filename = os.path.join(os.path.dirname(filename), sample_filename)

                with open(sample_filename, "rb") as f:
                    samples = np.fromfile(f, "<f4")
                assert (sample_count == len(samples))
                print("mean time: {:>8.6f} s, num_samples: {}".format(np.mean(samples),
                                                                       len(samples)))


if __name__ == '__main__':
    sys.exit(main())
