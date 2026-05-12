#!/usr/bin/env python
#
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

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cuda.bench.results import (
    BenchmarkResult,
    BenchmarkResultSummary,
    SubBenchmarkResult,
    SubBenchmarkState,
)


class MarkdownTable:
    def __init__(self):
        self.columns = []

    def add_cell(self, row: int, key: str, header: str, value: str) -> None:
        column = next((col for col in self.columns if col["key"] == key), None)
        if column is None:
            column = {
                "key": key,
                "header": header,
                "rows": [],
                "max_width": len(header),
            }
            self.columns.append(column)

        column["max_width"] = max(column["max_width"], len(value))
        while len(column["rows"]) <= row:
            column["rows"].append("")
        column["rows"][row] = value

    def to_string(self) -> str:
        if not self.columns:
            return ""

        num_rows = max(len(column["rows"]) for column in self.columns)
        for column in self.columns:
            while len(column["rows"]) < num_rows:
                column["rows"].append("")

        header = "|"
        divider = "|"
        for column in self.columns:
            width = column["max_width"]
            header += f" {column['header']:^{width}} |"
            divider += f"{'':-^{width + 2}}|"

        rows = []
        for row in range(num_rows):
            row_text = "|"
            for column in self.columns:
                row_text += f" {column['rows'][row]:>{column['max_width']}} |"
            rows.append(row_text)

        return "\n".join([header, divider, *rows]) + "\n"


def format_default(summary: BenchmarkResultSummary) -> str:
    value = summary.value
    if isinstance(value, float):
        return f"{value:.5g}"
    if value is None:
        return ""
    return str(value)


def format_duration(summary: BenchmarkResultSummary) -> str:
    seconds = float(summary["value"])
    if seconds >= 1.0:
        return f"{seconds:0.3f} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:0.3f} ms"
    if seconds >= 1e-6:
        return f"{seconds * 1e6:0.3f} us"
    return f"{seconds * 1e9:0.3f} ns"


def format_item_rate(summary: BenchmarkResultSummary) -> str:
    items_per_second = float(summary["value"])
    if items_per_second >= 1e15:
        return f"{items_per_second * 1e-15:0.3f}P"
    if items_per_second >= 1e12:
        return f"{items_per_second * 1e-12:0.3f}T"
    if items_per_second >= 1e9:
        return f"{items_per_second * 1e-9:0.3f}G"
    if items_per_second >= 1e6:
        return f"{items_per_second * 1e-6:0.3f}M"
    if items_per_second >= 1e3:
        return f"{items_per_second * 1e-3:0.3f}K"
    return f"{items_per_second:0.3f}"


def format_frequency(summary: BenchmarkResultSummary) -> str:
    frequency_hz = float(summary["value"])
    if frequency_hz >= 1e9:
        return f"{frequency_hz * 1e-9:0.3f} GHz"
    if frequency_hz >= 1e6:
        return f"{frequency_hz * 1e-6:0.3f} MHz"
    if frequency_hz >= 1e3:
        return f"{frequency_hz * 1e-3:0.3f} KHz"
    return f"{frequency_hz:0.3f} Hz"


def format_bytes(summary: BenchmarkResultSummary) -> str:
    nbytes = float(summary["value"])
    if nbytes >= 1024.0 * 1024.0 * 1024.0:
        return f"{nbytes / (1024.0 * 1024.0 * 1024.0):0.3f} GiB"
    if nbytes >= 1024.0 * 1024.0:
        return f"{nbytes / (1024.0 * 1024.0):0.3f} MiB"
    if nbytes >= 1024.0:
        return f"{nbytes / 1024.0:0.3f} KiB"
    return f"{nbytes:0.3f} B"


def format_byte_rate(summary: BenchmarkResultSummary) -> str:
    bytes_per_second = float(summary["value"])
    if bytes_per_second >= 1e15:
        return f"{bytes_per_second * 1e-15:0.3f} PB/s"
    if bytes_per_second >= 1e12:
        return f"{bytes_per_second * 1e-12:0.3f} TB/s"
    if bytes_per_second >= 1e9:
        return f"{bytes_per_second * 1e-9:0.3f} GB/s"
    if bytes_per_second >= 1e6:
        return f"{bytes_per_second * 1e-6:0.3f} MB/s"
    if bytes_per_second >= 1e3:
        return f"{bytes_per_second * 1e-3:0.3f} KB/s"
    return f"{bytes_per_second:0.3f} B/s"


def format_sample_size(summary: BenchmarkResultSummary) -> str:
    return f"{int(summary['value'])}x"


def format_percentage(summary: BenchmarkResultSummary) -> str:
    return f"{float(summary['value']) * 100.0:.2f}%"


def format_summary(summary: BenchmarkResultSummary) -> str:
    if summary.hint == "duration":
        return format_duration(summary)
    if summary.hint == "item_rate":
        return format_item_rate(summary)
    if summary.hint == "frequency":
        return format_frequency(summary)
    if summary.hint == "bytes":
        return format_bytes(summary)
    if summary.hint == "byte_rate":
        return format_byte_rate(summary)
    if summary.hint == "sample_size":
        return format_sample_size(summary)
    if summary.hint == "percentage":
        return format_percentage(summary)
    return format_default(summary)


def format_axis_value(
    axis_value: dict, axes_by_name: dict[str, dict]
) -> tuple[str, str]:
    name = axis_value["name"]
    axis = axes_by_name.get(name, {})
    value = axis_value["value"]
    if value is None:
        return name, ""

    if axis.get("type") == "int64" and axis.get("flags") == "pow2":
        int_value = int(value)
        exponent = int_value.bit_length() - 1
        return name, f"2^{exponent} = {int_value}"

    value_type = axis_value.get("type", axis.get("type"))
    if value_type == "int64":
        return name, str(int(value))
    if value_type == "float64":
        return name, f"{float(value):.5g}"

    return name, str(value)


def add_state_row(
    table: MarkdownTable,
    row: int,
    state: SubBenchmarkState,
    bench: SubBenchmarkResult,
) -> None:
    axes_by_name = {axis["name"]: axis for axis in bench.axes}

    for axis_value in state.axis_values:
        header, value = format_axis_value(axis_value, axes_by_name)
        table.add_cell(row, f"axis:{header}", header, value)

    for summary in state.summaries.values():
        if summary.hide is not None:
            continue
        header = summary.name if summary.name is not None else summary.tag
        table.add_cell(row, summary.tag, header, format_summary(summary))


def format_benchmark(result: BenchmarkResult, bench: SubBenchmarkResult) -> str:
    parts = [f"## {bench.name}\n\n"]
    device_ids: list[int | None] = list(bench.devices) if bench.devices else [None]

    for device_id in device_ids:
        if device_id is not None:
            device = result.devices.get(device_id)
            device_name = device.name if device is not None else f"Device {device_id}"
            parts.append(f"### [{device_id}] {device_name}\n\n")

        table = MarkdownTable()
        row = 0
        for state in bench.states:
            if device_id is not None and state.device != device_id:
                continue
            add_state_row(table, row, state, bench)
            row += 1

        table_text = table.to_string()
        parts.append(table_text if table_text else "No data -- check log.\n")

    return "".join(parts)


def format_result(result: BenchmarkResult) -> str:
    parts = ["# Benchmark Results\n"]
    for bench in result.values():
        parts.append(f"\n{format_benchmark(result, bench)}")
    return "".join(parts)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="nvbench-json-summary",
        description="Print an NVBench-style markdown summary table from NVBench JSON output.",
    )
    parser.add_argument("json_path", help="Path to an NVBench JSON output file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write markdown output to this file instead of stdout.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = BenchmarkResult.from_json(args.json_path)
    report = format_result(result)

    if args.output is not None:
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
