#!/usr/bin/env python
#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from string import Formatter
from typing import Any

if __package__:
    from .nvbench_tooling_deps import ToolingDependency, require_tooling_dependency
else:
    from nvbench_tooling_deps import ToolingDependency, require_tooling_dependency


PlotAlongData = dict[str, dict[str, dict[float, float | None]]]

PLOT_ALONG_OUTPUT_TEMPLATE_FIELDS = frozenset({"benchmark", "device", "axis", "pair"})
PLOT_OUTPUT_FIELD_SAFE_CHARS = re.compile(r"[^A-Za-z0-9_-]+")


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


def extract_plot_axis_value(axis_values, plot_along, benchmark_name, state_name):
    axis_name_parts = []
    for axis_value in axis_values:
        if axis_value["name"] != plot_along:
            axis_name_parts.append(f"""{axis_value["name"]} = {axis_value["value"]}""")
        else:
            return (
                parse_plot_axis_value(axis_value["name"], axis_value["value"]),
                axis_name_parts,
            )
    raise ValueError(
        f"--plot-along axis {plot_along!r} is not present in "
        f"benchmark {benchmark_name!r} state {state_name!r}"
    )


def format_plot_series_key(state_key, occurrence, occurrence_count, axis_name_parts):
    parts = []
    if state_key:
        parts.append(state_key)
    if occurrence_count > 1:
        parts.append(f"occurrence={occurrence + 1}/{occurrence_count}")
    parts.extend(axis_name_parts)
    return ", ".join(parts)


def ensure_plot_output_parent(output):
    output_path = Path(output)
    parent = output_path.parent
    if parent != Path("."):
        parent.mkdir(parents=True, exist_ok=True)


def reserve_plot_output_path(
    output: str | None,
    output_paths: set[str],
    *,
    option_name: str,
    collision_hint: str,
) -> None:
    if output is None:
        return

    normalized_output_path = os.path.abspath(output)
    if normalized_output_path in output_paths:
        raise ValueError(
            f"{option_name} would write multiple plots to {output!r}; {collision_hint}"
        )
    output_paths.add(normalized_output_path)


def validate_plot_along_output_template(output_template):
    try:
        parsed_fields = [
            (field_name, format_spec, conversion)
            for _, field_name, format_spec, conversion in Formatter().parse(
                output_template
            )
        ]
    except ValueError as exc:
        raise ValueError(f"--plot-along-output template is invalid: {exc}") from exc

    valid_fields = ", ".join(
        f"{{{field}}}" for field in sorted(PLOT_ALONG_OUTPUT_TEMPLATE_FIELDS)
    )
    for field_name, format_spec, conversion in parsed_fields:
        if field_name is None:
            continue
        if (
            field_name not in PLOT_ALONG_OUTPUT_TEMPLATE_FIELDS
            or format_spec
            or conversion
        ):
            raise ValueError(
                f"--plot-along-output supports template fields {valid_fields}; "
                f"got {{{field_name}}}"
            )


def sanitize_plot_output_component(value: object) -> str:
    sanitized = PLOT_OUTPUT_FIELD_SAFE_CHARS.sub("_", str(value))
    sanitized = sanitized.strip("._-")
    return sanitized or "value"


def format_plot_along_output_path(
    output_template, *, benchmark_name, device_id, axis_name, device_pair_index=0
):
    if output_template is None:
        return None

    validate_plot_along_output_template(output_template)
    try:
        return output_template.format(
            benchmark=sanitize_plot_output_component(benchmark_name),
            device=sanitize_plot_output_component(device_id),
            axis=sanitize_plot_output_component(axis_name),
            pair=sanitize_plot_output_component(device_pair_index),
        )
    except (IndexError, KeyError, ValueError) as exc:
        raise ValueError(f"--plot-along-output template is invalid: {exc}") from exc


def save_or_show_plot(fig, plt, output, description):
    if output is None:
        plt.show()
        return

    try:
        ensure_plot_output_parent(output)
        fig.savefig(output, dpi=150)
    except OSError as exc:
        raise ValueError(f"failed to write {description} to {output!r}: {exc}") from exc
    print(f"Saved {description} to {output}")


def use_noninteractive_matplotlib_backend(matplotlib):
    matplotlib.use("Agg")


def plot_comparison_entries(
    entries, title=None, dark=False, output=None, *, tool_name="nvbench-compare-robust"
):
    if not entries:
        print("No comparison data to plot.")
        return 1

    matplotlib = require_tooling_dependency(
        ToolingDependency("matplotlib", "matplotlib", "plot rendering", extra="plot"),
        tool_name=tool_name,
    )
    if output is not None:
        use_noninteractive_matplotlib_backend(matplotlib)

    plt = require_tooling_dependency(
        ToolingDependency(
            "matplotlib.pyplot", "matplotlib", "plot rendering", extra="plot"
        ),
        tool_name=tool_name,
    )
    ticker = require_tooling_dependency(
        ToolingDependency(
            "matplotlib.ticker", "matplotlib", "plot axis formatting", extra="plot"
        ),
        tool_name=tool_name,
    )
    PercentFormatter = ticker.PercentFormatter

    labels, values, statuses, bench_names = map(list, zip(*entries, strict=True))

    status_colors = {
        "SLOW": "red",
        "FAST": "green",
        "SAME": "blue",
    }
    colors = [status_colors.get(status, "gray") for status in statuses]

    fig_height = max(4.0, 0.3 * len(entries) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    try:
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

        save_or_show_plot(fig, plt, output, "comparison plot")
    finally:
        plt.close(fig)
    return 0


def make_plot_along_data() -> PlotAlongData:
    return {
        "cmp": {},
        "ref": {},
        "cmp_noise": {},
        "ref_noise": {},
    }


def has_plot_along_data(plot_data: PlotAlongData) -> bool:
    return any(axis_times for axis_times in plot_data["cmp"].values())


def is_positive_finite(value):
    return value is not None and math.isfinite(value) and value > 0.0


def is_usable_noise(value):
    return value is not None and math.isfinite(value) and value >= 0.0


@dataclass
class PlotCollector:
    plot_along: str | None
    plot_summary: bool
    dark: bool
    plot_output: str | None
    plot_along_output: str | None
    output_paths: set[str]
    tool_name: str
    comparison_entries: list[tuple[str, float, str, str]] = field(default_factory=list)
    comparison_device_names: set[str] = field(default_factory=set)
    plt: Any = None

    def __post_init__(self) -> None:
        if self.plot_along:
            if self.plot_along_output is not None:
                matplotlib = require_tooling_dependency(
                    ToolingDependency(
                        "matplotlib",
                        "matplotlib",
                        "per-axis plot rendering",
                        extra="plot",
                    ),
                    tool_name=self.tool_name,
                )
                use_noninteractive_matplotlib_backend(matplotlib)
            self.plt = require_tooling_dependency(
                ToolingDependency(
                    "matplotlib.pyplot",
                    "matplotlib",
                    "per-axis plot rendering",
                    extra="plot",
                ),
                tool_name=self.tool_name,
            )
            sns = require_tooling_dependency(
                ToolingDependency(
                    "seaborn", "seaborn", "per-axis plot styling", extra="plot"
                ),
                tool_name=self.tool_name,
            )

            sns.set_theme()

    def make_plot_along_data(self) -> PlotAlongData:
        return make_plot_along_data()

    def record_plot_along(
        self,
        plot_data: PlotAlongData,
        *,
        ref_time,
        cmp_time,
        ref_noise,
        cmp_noise,
        axis_values,
        benchmark_name,
        state_name,
        occurrence,
        occurrence_count,
    ) -> None:
        if (
            self.plot_along is None
            or not is_positive_finite(ref_time)
            or not is_positive_finite(cmp_time)
        ):
            return

        axis_value, axis_name_parts = extract_plot_axis_value(
            axis_values, self.plot_along, benchmark_name, state_name
        )
        axis_name = format_plot_series_key(
            state_name,
            occurrence,
            occurrence_count,
            axis_name_parts,
        )

        if axis_name not in plot_data["cmp"]:
            plot_data["cmp"][axis_name] = {}
            plot_data["ref"][axis_name] = {}
            plot_data["cmp_noise"][axis_name] = {}
            plot_data["ref_noise"][axis_name] = {}

        plot_data["cmp"][axis_name][axis_value] = cmp_time
        plot_data["ref"][axis_name][axis_value] = ref_time
        plot_data["cmp_noise"][axis_name][axis_value] = cmp_noise
        plot_data["ref_noise"][axis_name][axis_value] = ref_noise

    def has_plot_along_data(self, plot_data: PlotAlongData) -> bool:
        return bool(self.plot_along) and has_plot_along_data(plot_data)

    def record_summary_entry(
        self, *, benchmark_name, axis_label, cmp_device_name, frac_diff, status
    ) -> None:
        if not self.plot_summary or frac_diff is None or not math.isfinite(frac_diff):
            return

        if axis_label:
            label = f"{benchmark_name} | {axis_label}"
        else:
            label = benchmark_name
        if cmp_device_name:
            self.comparison_device_names.add(cmp_device_name)
        self.comparison_entries.append((label, frac_diff, status, benchmark_name))

    def render_plot_along(
        self,
        plot_data: PlotAlongData,
        *,
        benchmark_name,
        cmp_device_id,
        cmp_device_index,
        cmp_device_name,
    ) -> None:
        if self.plot_along is None or self.plt is None:
            return

        plot_along_output_path = format_plot_along_output_path(
            self.plot_along_output,
            benchmark_name=benchmark_name,
            device_id=cmp_device_id,
            axis_name=self.plot_along,
            device_pair_index=cmp_device_index,
        )
        if plot_along_output_path is not None:
            reserve_plot_output_path(
                plot_along_output_path,
                self.output_paths,
                option_name="--plot-along-output",
                collision_hint="use a template that yields a unique path for each plot",
            )

        fig = self.plt.figure()
        try:
            self.plt.xscale("log")
            self.plt.yscale("log")
            self.plt.xlabel(self.plot_along)
            self.plt.ylabel("time [s]")
            self.plt.title(cmp_device_name)

            for axis in plot_data["cmp"].keys():
                self._plot_line(plot_data, "cmp", "-", axis, axis)
                self._plot_line(plot_data, "ref", "--", axis + " ref", axis)

            self.plt.legend()
            save_or_show_plot(
                fig, self.plt, plot_along_output_path, "plot-along output"
            )
        finally:
            self.plt.close(fig)

    def _plot_line(self, plot_data: PlotAlongData, key, shape, label, data_axis):
        axis_times = plot_data[key][data_axis]
        if not axis_times:
            return
        axis_noise = plot_data[key + "_noise"][data_axis]
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

        p = self.plt.plot(x, y, shape, marker="o", label=label)

        def plot_confidence_band(first, last):
            if last - first < 2:
                return

            band_x = x[first:last]
            band_y = y[first:last]
            band_noise = noise[first:last]
            top = [band_y[i] + band_y[i] * band_noise[i] for i in range(len(band_x))]
            bottom = [
                max(
                    band_y[i] - band_y[i] * band_noise[i],
                    band_y[i] * 0.001,
                )
                for i in range(len(band_x))
            ]
            self.plt.fill_between(
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

    def render_summary(self, global_axis_filters) -> None:
        if not self.plot_summary:
            return

        title = "GPU timing change"
        if len(self.comparison_device_names) == 1:
            title = f"{title} - {next(iter(self.comparison_device_names))}"
        if global_axis_filters:
            axis_label = ", ".join(
                axis_filter["display"]
                for axis_filter in global_axis_filters
                if len(axis_filter["values"]) == 1
            )
            if axis_label:
                title = f"{title} ({axis_label})"
        if self.comparison_entries:
            reserve_plot_output_path(
                self.plot_output,
                self.output_paths,
                option_name="--plot-output",
                collision_hint=(
                    "use a unique output path for each input pair or compare one "
                    "JSON pair at a time"
                ),
            )
        plot_comparison_entries(
            self.comparison_entries,
            title=title,
            dark=self.dark,
            output=self.plot_output,
            tool_name=self.tool_name,
        )
