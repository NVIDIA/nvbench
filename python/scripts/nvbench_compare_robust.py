#!/usr/bin/env python
#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import argparse
import math
import os
import pprint
import re
import sys
import warnings
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING, Any, BinaryIO, Callable, Protocol, TypeAlias

if TYPE_CHECKING:
    import numpy as _np
    from numpy.typing import NDArray

    Float32Array: TypeAlias = NDArray[_np.float32]
    NumpyArray: TypeAlias = NDArray[Any]
else:
    Float32Array: TypeAlias = Any
    NumpyArray: TypeAlias = Any

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


def current_tool_name() -> str:
    return os.path.basename(sys.argv[0]) or "nvbench-compare-robust"


np: Any = None
Fore: Any = None


def load_nvbench_compare_tooling(*, load_color: bool = True) -> None:
    global Fore, np

    if np is None:
        np = require_tooling_dependency(
            ToolingDependency(
                "numpy", "numpy", "bulk timing analysis", extra="compare"
            ),
            tool_name=current_tool_name(),
        )
    if load_color and Fore is None:
        colorama = require_tooling_dependency(
            ToolingDependency(
                "colorama", "colorama", "colored status output", extra="compare"
            ),
            tool_name=current_tool_name(),
        )
        Fore = colorama.Fore


def load_tabulate_for_table_output() -> tuple[Any, tuple[int, ...]]:
    tabulate_module = require_tooling_dependency(
        ToolingDependency("tabulate", "tabulate", "table output", extra="compare"),
        tool_name=current_tool_name(),
    )
    return tabulate_module, version_tuple(tabulate_module.__version__)


def load_jsondiff_for_device_diff() -> Any:
    return require_tooling_dependency(
        ToolingDependency(
            "jsondiff", "jsondiff", "device metadata diffs", extra="compare"
        ),
        tool_name=current_tool_name(),
    )


GPU_TIME_MIN_TAG = "nv/cold/time/gpu/min"
GPU_TIME_MAX_TAG = "nv/cold/time/gpu/max"
GPU_TIME_MEAN_TAG = "nv/cold/time/gpu/mean"
GPU_TIME_STDEV_TAG = "nv/cold/time/gpu/stdev/absolute"
GPU_TIME_STDEV_RELATIVE_TAG = "nv/cold/time/gpu/stdev/relative"
GPU_TIME_Q1_TAG = "nv/cold/time/gpu/q1"
GPU_TIME_MEDIAN_TAG = "nv/cold/time/gpu/median"
GPU_TIME_Q3_TAG = "nv/cold/time/gpu/q3"
GPU_TIME_IQR_TAG = "nv/cold/time/gpu/iqr/absolute"
GPU_TIME_IQR_RELATIVE_TAG = "nv/cold/time/gpu/iqr/relative"
LEGACY_GPU_TIME_IR_TAG = "nv/cold/time/gpu/ir/absolute"
LEGACY_GPU_TIME_IR_RELATIVE_TAG = "nv/cold/time/gpu/ir/relative"
GPU_SM_CLOCK_RATE_MEAN_TAG = "nv/cold/sm_clock_rate/mean"
SAMPLE_TIMES_TAG = "nv/json/bin:nv/cold/sample_times"
SAMPLE_FREQUENCIES_TAG = "nv/json/freqs-bin:nv/cold/sample_freqs"

# The reader returns an object supporting the buffer protocol. Python 3.10 does
# not provide a standard Buffer type annotation.
Float32Reader = Callable[[str], object]


class TomlModule(Protocol):
    # TOML support is imported lazily. This protocol documents the narrow
    # tomllib/tomli module surface used by this script.
    @property
    def TOMLDecodeError(self) -> type[BaseException]: ...

    def load(self, fp: BinaryIO, /) -> dict[str, Any]: ...


def read_float32_file(filename: str) -> object:
    return np.fromfile(filename, dtype="<f4")


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


# These dataclasses are treated as parsed value objects. frozen=True prevents
# accidental field reassignment but does not imply deep immutability.


@dataclass(frozen=True)
class ComparisonThresholds:
    clear_gap_relative: float
    same_center_relative: float
    same_overlap_fraction: float
    same_relative_dispersion_ceiling: float
    bulk_same_sample_coverage: float
    bulk_same_support_coverage: float
    bulk_support_rare_sample_fraction: float
    bulk_support_max_removed_sample_fraction: float


COMPARISON_THRESHOLD_PRESET_VALUES = {
    "default": {
        "clear_gap_relative": 0.005,
        "same_center_relative": 0.005,
        "same_overlap_fraction": 0.5,
        "same_relative_dispersion_ceiling": 0.02,
        "bulk_same_sample_coverage": 0.97,
        "bulk_same_support_coverage": 0.80,
        "bulk_support_rare_sample_fraction": 0.001,
        "bulk_support_max_removed_sample_fraction": 0.01,
    },
    "strict": {
        "clear_gap_relative": 0.01,
        "same_center_relative": 0.0025,
        "same_overlap_fraction": 0.75,
        "same_relative_dispersion_ceiling": 0.01,
        "bulk_same_sample_coverage": 0.995,
        "bulk_same_support_coverage": 0.90,
        "bulk_support_rare_sample_fraction": 0.001,
        "bulk_support_max_removed_sample_fraction": 0.005,
    },
    "permissive": {
        "clear_gap_relative": 0.0025,
        "same_center_relative": 0.01,
        "same_overlap_fraction": 0.25,
        "same_relative_dispersion_ceiling": 0.05,
        "bulk_same_sample_coverage": 0.90,
        "bulk_same_support_coverage": 0.60,
        "bulk_support_rare_sample_fraction": 0.001,
        "bulk_support_max_removed_sample_fraction": 0.02,
    },
}

COMPARISON_THRESHOLD_PRESETS = {
    name: ComparisonThresholds(**values)
    for name, values in COMPARISON_THRESHOLD_PRESET_VALUES.items()
}

COMPARISON_CONFIG_VERSION = 1
COMPARISON_DEFAULT_PRESET = "default"
COMPARISON_CONFIG_TABLES = {
    "preset",
    "clear_gap",
    "same",
    "bulk",
}
COMPARISON_CONFIG_KEYS = {
    "clear_gap": {
        "relative": "clear_gap_relative",
    },
    "same": {
        "center_relative": "same_center_relative",
        "overlap_fraction": "same_overlap_fraction",
        "relative_dispersion_ceiling": "same_relative_dispersion_ceiling",
    },
    "bulk": {
        "sample_coverage": "bulk_same_sample_coverage",
        "support_coverage": "bulk_same_support_coverage",
    },
    "bulk.rare_support": {
        "sample_fraction": "bulk_support_rare_sample_fraction",
        "max_removed_sample_fraction": "bulk_support_max_removed_sample_fraction",
    },
}
COMPARISON_THRESHOLD_RANGES = {
    "clear_gap_relative": (0.0, None),
    "same_center_relative": (0.0, None),
    "same_overlap_fraction": (0.0, 1.0),
    "same_relative_dispersion_ceiling": (0.0, None),
    "bulk_same_sample_coverage": (0.0, 1.0),
    "bulk_same_support_coverage": (0.0, 1.0),
    "bulk_support_rare_sample_fraction": (0.0, 1.0),
    "bulk_support_max_removed_sample_fraction": (0.0, 1.0),
}


def get_default_thresholds() -> ComparisonThresholds:
    return COMPARISON_THRESHOLD_PRESETS[COMPARISON_DEFAULT_PRESET]


def get_comparison_thresholds(preset_name: str) -> ComparisonThresholds:
    try:
        return COMPARISON_THRESHOLD_PRESETS[preset_name]
    except KeyError as exc:
        raise ValueError(f"unknown comparison preset {preset_name!r}") from exc


def load_toml_module() -> TomlModule:
    try:
        # built-in Python module, added in 3.11 via PEP 680
        import tomllib

        return tomllib
    except ModuleNotFoundError:
        try:
            # third-party library for Python 3.10
            # note Python 3.10 EOL date is Oct. 31, 2026
            import tomli

            return tomli
        except ModuleNotFoundError as exc:
            raise ValueError(
                "TOML config support requires Python 3.11+ or the tomli package"
            ) from exc


def validate_config_table(value: object, table_name: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"config table [{table_name}] must be a TOML table")


def validate_config_float(value: object, key: str, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"config value {key!r} must be a finite number")

    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"config value {key!r} must be finite")

    minimum, maximum = COMPARISON_THRESHOLD_RANGES[field_name]
    if value < minimum:
        raise ValueError(f"config value {key!r} must be >= {minimum:g}")
    if maximum is not None and value > maximum:
        raise ValueError(f"config value {key!r} must be <= {maximum:g}")
    return value


def parse_config_section(
    table: Mapping[str, Any], section_name: str
) -> dict[str, float]:
    validate_config_table(table, section_name)
    known_keys = COMPARISON_CONFIG_KEYS[section_name]
    unknown_keys = set(table) - set(known_keys)
    if unknown_keys:
        unknown = ", ".join(sorted(unknown_keys))
        raise ValueError(f"unknown config key(s) in [{section_name}]: {unknown}")

    overrides = {}
    for key, field_name in known_keys.items():
        if key not in table:
            continue
        full_key = f"{section_name}.{key}"
        overrides[field_name] = validate_config_float(table[key], full_key, field_name)
    return overrides


def parse_comparison_config_data(
    config_data: Mapping[str, Any],
) -> tuple[str | None, dict[str, float]]:
    if not isinstance(config_data, Mapping):
        raise ValueError("comparison config must be a TOML table")

    unknown_top_level = set(config_data) - ({"version"} | COMPARISON_CONFIG_TABLES)
    if unknown_top_level:
        unknown = ", ".join(sorted(unknown_top_level))
        raise ValueError(f"unknown top-level config key(s): {unknown}")

    version = config_data.get("version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError(
            f"comparison config must specify integer version = {COMPARISON_CONFIG_VERSION}"
        )
    if version != COMPARISON_CONFIG_VERSION:
        raise ValueError(
            f"unsupported comparison config version {version!r}; "
            f"expected {COMPARISON_CONFIG_VERSION}"
        )

    preset_name = None
    if "preset" in config_data:
        preset_table = config_data["preset"]
        validate_config_table(preset_table, "preset")
        unknown_keys = set(preset_table) - {"name"}
        if unknown_keys:
            unknown = ", ".join(sorted(unknown_keys))
            raise ValueError(f"unknown config key(s) in [preset]: {unknown}")
        if "name" in preset_table:
            preset_name = preset_table["name"]
            if not isinstance(preset_name, str):
                raise ValueError("config value 'preset.name' must be a string")
            get_comparison_thresholds(preset_name)

    overrides = {}
    for section_name in ("clear_gap", "same"):
        if section_name in config_data:
            overrides.update(
                parse_config_section(config_data[section_name], section_name)
            )

    if "bulk" in config_data:
        bulk_table = config_data["bulk"]
        validate_config_table(bulk_table, "bulk")
        known_bulk_keys = set(COMPARISON_CONFIG_KEYS["bulk"]) | {"rare_support"}
        unknown_keys = set(bulk_table) - known_bulk_keys
        if unknown_keys:
            unknown = ", ".join(sorted(unknown_keys))
            raise ValueError(f"unknown config key(s) in [bulk]: {unknown}")

        bulk_values = {
            key: value for key, value in bulk_table.items() if key != "rare_support"
        }
        overrides.update(parse_config_section(bulk_values, "bulk"))
        if "rare_support" in bulk_table:
            overrides.update(
                parse_config_section(bulk_table["rare_support"], "bulk.rare_support")
            )

    return preset_name, overrides


def read_comparison_config_file(
    config_path: str | os.PathLike[str],
) -> tuple[str | None, dict[str, float]]:
    toml_module = load_toml_module()
    try:
        with open(config_path, "rb") as config_file:
            config_data = toml_module.load(config_file)
    except toml_module.TOMLDecodeError as exc:
        raise ValueError(
            f"failed to parse comparison config {config_path!r}: {exc}"
        ) from exc
    except OSError as exc:
        raise ValueError(
            f"failed to read comparison config {config_path!r}: {exc}"
        ) from exc

    return parse_comparison_config_data(config_data)


def resolve_comparison_thresholds(
    cli_preset_name: str | None = None,
    config_path: str | os.PathLike[str] | None = None,
) -> tuple[str, ComparisonThresholds]:
    config_preset_name = None
    config_overrides: dict[str, float] = {}
    if config_path is not None:
        config_preset_name, config_overrides = read_comparison_config_file(config_path)

    preset_name = cli_preset_name or config_preset_name or COMPARISON_DEFAULT_PRESET
    thresholds = replace(get_comparison_thresholds(preset_name), **config_overrides)
    return preset_name, thresholds


def format_toml_float(value: float) -> str:
    return repr(float(value))


def dump_comparison_config(preset_name: str, thresholds: ComparisonThresholds) -> str:
    lines = [
        f"version = {COMPARISON_CONFIG_VERSION}",
        "",
        "[preset]",
        f'name = "{preset_name}"',
        "",
        "[clear_gap]",
        f"relative = {format_toml_float(thresholds.clear_gap_relative)}",
        "",
        "[same]",
        f"center_relative = {format_toml_float(thresholds.same_center_relative)}",
        f"overlap_fraction = {format_toml_float(thresholds.same_overlap_fraction)}",
        "relative_dispersion_ceiling = "
        f"{format_toml_float(thresholds.same_relative_dispersion_ceiling)}",
        "",
        "[bulk]",
        f"sample_coverage = {format_toml_float(thresholds.bulk_same_sample_coverage)}",
        f"support_coverage = {format_toml_float(thresholds.bulk_same_support_coverage)}",
        "",
        "[bulk.rare_support]",
        "sample_fraction = "
        f"{format_toml_float(thresholds.bulk_support_rare_sample_fraction)}",
        "max_removed_sample_fraction = "
        f"{format_toml_float(thresholds.bulk_support_max_removed_sample_fraction)}",
    ]
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class SupportFilterInfo:
    activated: bool
    reason: str
    removed_sample_fraction: float


@dataclass(frozen=True)
class Float32BinarySource:
    count: int
    filename: str
    json_dir: str
    description: str
    json_path: str | None = None
    reader: Float32Reader = read_float32_file

    @cached_property
    def values(self) -> Float32Array | None:
        return read_float32_binary(
            self.count,
            self.filename,
            self.json_dir,
            self.description,
            self.reader,
            self.json_path,
        )

    def has_material_payload(self) -> bool:
        if self.count <= 0:
            return False

        if self.reader is not read_float32_file:
            return True

        filename = resolve_binary_filename(self.json_dir, self.filename, self.json_path)
        try:
            return os.path.getsize(filename) > 0
        except OSError:
            return False


@dataclass(frozen=True)
class GpuTimingData:
    minimum: float | None
    maximum: float | None
    mean: float | None
    stdev: float | None
    stdev_relative: float | None
    first_quartile: float | None
    median: float | None
    third_quartile: float | None
    interquartile_range: float | None
    interquartile_range_relative: float | None
    sm_clock_rate_mean: float | None = None
    sample_source: Float32BinarySource | None = None
    frequency_source: Float32BinarySource | None = None

    @cached_property
    def samples(self) -> Float32Array | None:
        if self.sample_source is None:
            return None
        return self.sample_source.values

    @cached_property
    def frequencies(self) -> Float32Array | None:
        if self.frequency_source is None:
            return None
        return self.frequency_source.values


@dataclass(frozen=True)
class BulkDebugOutput:
    destination: str

    @property
    def is_stdout(self) -> bool:
        return self.destination.lower() == "stdout"


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
    UNDECIDED = "AMBG"
    SAME = "SAME"
    FAST = "FAST"
    SLOW = "SLOW"


@dataclass(frozen=True)
class DecisionReason:
    code: str
    message: str
    severity: float = 0.0


REASON_DISPLAY_CODES = {
    "bulk_cycle_data_unusable": "bc-bad",
    "bulk_cycle_gap_not_confirmed": "bc-gap-miss",
    "bulk_cycle_same": "bc-same",
    "bulk_cycle_support_mismatch": "bc-sup-miss",
    "bulk_data_unavailable": "bulk-miss",
    "bulk_same": "bulk-same",
    "bulk_time_data_unusable": "bt-bad",
    "bulk_time_same": "bt-same",
    "bulk_time_same_confirmed_by_summary_cycles": "bt-same-sc",
    "bulk_time_same_without_cycles": "bt-same-no-cyc",
    "bulk_time_support_mismatch": "bt-sup-miss",
    "centers_not_close": "centers-far",
    "clear_gap_confirmed_by_bulk_cycles": "bc-gap",
    "clear_gap_confirmed_by_summary_cycles": "sc-gap",
    "cycle_same_not_confirmed": "sc-same-miss",
    "invalid_clock_rate": "clk-bad",
    "missing_clock_rate": "clk-miss",
    "missing_interval": "int-miss",
    "no_clear_gap": "no-gap",
    "noise_too_high": "noise-high",
    "noise_unavailable": "noise-miss",
    "same_confirmed_by_cycles": "sc-same",
    "same_summary": "sum-same",
    "same_without_clock_rate": "same-no-clk",
    "summary_cycle_gap_not_confirmed": "sc-gap-miss",
    "gpu_timing_summaries_missing": "summ-miss",
    "state_skipped": "state-skip",
    "timing_center_missing": "center-miss",
    "timing_center_nonfinite": "center-nonfin",
    "timing_center_nonpositive": "center-nonpos",
    "weak_interval_overlap": "weak-overlap",
}


def format_reason_display_code(code):
    return REASON_DISPLAY_CODES.get(code, code)


def format_reason_legend_entries(reason_legend):
    entries = []
    for code, reason_summary in sorted(reason_legend.items()):
        if code == reason_summary.canonical_code.replace("_", "-"):
            continue
        entries.append(f"{code} = {reason_summary.canonical_code}")
    return entries


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
    max_noise: float | None
    status: ComparisonStatus
    reason: DecisionReason


@dataclass
class DecisionReasonSummary:
    count: int = 0
    canonical_code: str = ""
    message: str = ""
    severity: float = 0.0


@dataclass
class ComparisonStats:
    config_count: int = 0
    pass_count: int = 0
    improvement_count: int = 0
    regression_count: int = 0
    undecided_count: int = 0
    unknown_count: int = 0
    undecided_reasons: dict[str, DecisionReasonSummary] = field(default_factory=dict)
    reason_legend: dict[str, DecisionReasonSummary] = field(default_factory=dict)

    @staticmethod
    def record_reason_summary(
        summaries: dict[str, DecisionReasonSummary],
        reason: DecisionReason,
        *,
        use_display_code,
    ) -> None:
        display_code = (
            format_reason_display_code(reason.code) if use_display_code else reason.code
        )
        summary = summaries.setdefault(
            display_code, DecisionReasonSummary(canonical_code=reason.code)
        )
        if summary.count == 0 or reason.severity > summary.severity:
            summary.canonical_code = reason.code
            summary.message = reason.message
            summary.severity = reason.severity
        summary.count += 1

    def record(
        self, status: ComparisonStatus, reason: DecisionReason | None = None
    ) -> None:
        self.config_count += 1
        if reason is not None:
            self.record_reason_summary(
                self.reason_legend, reason, use_display_code=True
            )
        if status == ComparisonStatus.UNKNOWN:
            self.unknown_count += 1
        elif status == ComparisonStatus.UNDECIDED:
            self.undecided_count += 1
            if reason is not None:
                self.record_reason_summary(
                    self.undecided_reasons, reason, use_display_code=False
                )
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


def validate_threshold_diff(threshold):
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("--threshold-diff must be a finite non-negative percentage")


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


def extract_summary_float_with_fallback(
    summaries: list[dict[str, Any]],
    primary_tag: str,
    fallback_tag: str,
    *,
    null_value: float | None = None,
) -> float | None:
    value = extract_summary_float(summaries, primary_tag, null_value=null_value)
    if value is not None:
        return value
    return extract_summary_float(summaries, fallback_tag, null_value=null_value)


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
    if isinstance(value, bool):
        raise ValueError(
            f"summary {summary.get('tag', '<unknown>')!r} field 'size' "
            f"value {value!r} is not an int64"
        )
    if isinstance(value, float):
        raise ValueError(
            f"summary {summary.get('tag', '<unknown>')!r} field 'size' "
            f"value {value!r} is not an int64"
        )
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


def resolve_legacy_jsonbin_filename(json_dir, binary_filename, json_path):
    if json_path is None:
        return None

    json_name = os.path.basename(json_path)
    sidecar_names = {f"{json_name}-bin", f"{json_name}-freqs-bin"}
    path_parts = re.split(r"[\\/]+", os.path.normpath(binary_filename))
    for index, part in enumerate(path_parts):
        if part in sidecar_names:
            candidate = os.path.join(json_dir, *path_parts[index:])
            if os.path.exists(candidate):
                return candidate

    return None


def resolve_binary_filename(json_dir, binary_filename, json_path=None):
    if os.path.isabs(binary_filename):
        return binary_filename

    json_relative_filename = os.path.join(json_dir, binary_filename)
    if os.path.exists(json_relative_filename):
        return json_relative_filename

    legacy_jsonbin_filename = resolve_legacy_jsonbin_filename(
        json_dir, binary_filename, json_path
    )
    if legacy_jsonbin_filename is not None:
        return legacy_jsonbin_filename

    parent_relative_filename = os.path.join(os.path.dirname(json_dir), binary_filename)
    if os.path.exists(parent_relative_filename):
        return parent_relative_filename

    return json_relative_filename


def warn_unavailable_bulk_data(description, message):
    warnings.warn(
        f"Could not use NVBench {description} data: {message}; treating it as unavailable",
        RuntimeWarning,
        stacklevel=3,
    )


def read_float32_binary(count, filename, json_dir, description, reader, json_path=None):
    filename = resolve_binary_filename(json_dir, filename, json_path)
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


def extract_float32_binary_source(
    summaries, tag, json_dir, description, reader, json_path=None
):
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
        json_path=json_path,
        description=description,
        reader=reader,
    )


def extract_sample_time_source(summaries, json_dir, reader, json_path=None):
    return extract_float32_binary_source(
        summaries, SAMPLE_TIMES_TAG, json_dir, "sample time", reader, json_path
    )


def extract_sample_frequency_source(summaries, json_dir, reader, json_path=None):
    return extract_float32_binary_source(
        summaries,
        SAMPLE_FREQUENCIES_TAG,
        json_dir,
        "sample frequency",
        reader,
        json_path,
    )


def extract_gpu_timing_data(
    summaries, json_dir=None, float32_reader=read_float32_file, json_path=None
):
    sample_source = extract_sample_time_source(
        summaries, json_dir, float32_reader, json_path
    )
    frequency_source = extract_sample_frequency_source(
        summaries, json_dir, float32_reader, json_path
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

    mean = extract_summary_float(summaries, GPU_TIME_MEAN_TAG)
    stdev = extract_summary_float(summaries, GPU_TIME_STDEV_TAG, null_value=math.inf)
    stdev_relative = extract_summary_float(
        summaries, GPU_TIME_STDEV_RELATIVE_TAG, null_value=math.inf
    )
    if stdev is None:
        stdev = derive_absolute_dispersion(stdev_relative, mean)

    return GpuTimingData(
        minimum=extract_summary_float(summaries, GPU_TIME_MIN_TAG),
        maximum=extract_summary_float(summaries, GPU_TIME_MAX_TAG),
        mean=mean,
        stdev=stdev,
        stdev_relative=stdev_relative,
        first_quartile=extract_summary_float(summaries, GPU_TIME_Q1_TAG),
        median=extract_summary_float(summaries, GPU_TIME_MEDIAN_TAG),
        third_quartile=extract_summary_float(summaries, GPU_TIME_Q3_TAG),
        interquartile_range=extract_summary_float_with_fallback(
            summaries,
            GPU_TIME_IQR_TAG,
            LEGACY_GPU_TIME_IR_TAG,
            null_value=math.inf,
        ),
        interquartile_range_relative=extract_summary_float_with_fallback(
            summaries,
            GPU_TIME_IQR_RELATIVE_TAG,
            LEGACY_GPU_TIME_IR_RELATIVE_TAG,
            null_value=math.inf,
        ),
        sm_clock_rate_mean=extract_summary_float(summaries, GPU_SM_CLOCK_RATE_MEAN_TAG),
        sample_source=sample_source,
        frequency_source=frequency_source,
    )


def make_empty_gpu_timing_data():
    return GpuTimingData(
        minimum=None,
        maximum=None,
        mean=None,
        stdev=None,
        stdev_relative=None,
        first_quartile=None,
        median=None,
        third_quartile=None,
        interquartile_range=None,
        interquartile_range_relative=None,
    )


def resolve_bulk_source_filename(source: Float32BinarySource | None) -> str | None:
    if source is None:
        return None
    return os.path.abspath(
        resolve_binary_filename(source.json_dir, source.filename, source.json_path)
    )


def get_bulk_source_count(source: Float32BinarySource | None) -> int | None:
    if source is None:
        return None
    return source.count


def make_axis_debug_values(axis_values, axes) -> list[dict[str, Any]]:
    return [
        {
            "name": axis_value.get("name"),
            "type": axis_value.get("type"),
            "value": axis_value.get("value"),
            "display": format_axis_value(axis_value["name"], axis_value, axes),
        }
        for axis_value in axis_values
    ]


def make_bulk_debug_row(
    *,
    row_index: int,
    table_row_index: int,
    benchmark_name: str,
    ref_json_path: str | None,
    cmp_json_path: str | None,
    ref_device_id: int,
    cmp_device_id: int,
    cmp_state_name: str,
    occurrence: int,
    occurrence_count: int,
    axis_values,
    axes,
    ref_timing: GpuTimingData,
    cmp_timing: GpuTimingData,
    comparison: SummaryComparison,
) -> dict[str, Any]:
    return {
        "row_index": row_index,
        "table_row_index": table_row_index,
        "benchmark": benchmark_name,
        "reference_json": ref_json_path,
        "compare_json": cmp_json_path,
        "reference_device_id": ref_device_id,
        "compare_device_id": cmp_device_id,
        "state_key": cmp_state_name,
        "occurrence": occurrence,
        "occurrence_count": occurrence_count,
        "axis_values": make_axis_debug_values(axis_values, axes),
        "status": comparison.status.value,
        "reason": comparison.reason.code,
        "reason_message": comparison.reason.message,
        "reference_time": comparison.ref_time,
        "compare_time": comparison.cmp_time,
        "fractional_difference": comparison.frac_diff,
        "reference_sample_filename": resolve_bulk_source_filename(
            ref_timing.sample_source
        ),
        "reference_sample_count": get_bulk_source_count(ref_timing.sample_source),
        "reference_frequency_filename": resolve_bulk_source_filename(
            ref_timing.frequency_source
        ),
        "reference_frequency_count": get_bulk_source_count(ref_timing.frequency_source),
        "compare_sample_filename": resolve_bulk_source_filename(
            cmp_timing.sample_source
        ),
        "compare_sample_count": get_bulk_source_count(cmp_timing.sample_source),
        "compare_frequency_filename": resolve_bulk_source_filename(
            cmp_timing.frequency_source
        ),
        "compare_frequency_count": get_bulk_source_count(cmp_timing.frequency_source),
    }


def format_bulk_debug_python(bulk_rows: list[dict[str, Any]]) -> str:
    return (
        "# NVB-BULK-BEGIN\n"
        "# Generated by nvbench-compare-robust --bulk-debug-python.\n"
        "import numpy as np\n\n"
        "# pprint emits bare nan/inf tokens for non-finite floats.\n"
        "# Define them so this generated script remains executable.\n"
        'nan = float("nan")\n'
        'inf = float("inf")\n\n'
        f"bulk_rows = {pprint.pformat(bulk_rows, sort_dicts=False)}\n\n"
        "def read_float32(filename, expected_count=None):\n"
        "    if filename is None:\n"
        "        return None\n"
        "    values = np.fromfile(filename, dtype='<f4')\n"
        "    if expected_count is not None and len(values) != expected_count:\n"
        "        raise ValueError(\n"
        "            f'{filename!r}: expected {expected_count} float32 values, '\n"
        "            f'found {len(values)}'\n"
        "        )\n"
        "    return values\n\n"
        "def load_bulk_data(row):\n"
        "    return {\n"
        "        'reference_samples': read_float32(\n"
        "            row['reference_sample_filename'], row['reference_sample_count']\n"
        "        ),\n"
        "        'reference_frequencies': read_float32(\n"
        "            row['reference_frequency_filename'], row['reference_frequency_count']\n"
        "        ),\n"
        "        'compare_samples': read_float32(\n"
        "            row['compare_sample_filename'], row['compare_sample_count']\n"
        "        ),\n"
        "        'compare_frequencies': read_float32(\n"
        "            row['compare_frequency_filename'], row['compare_frequency_count']\n"
        "        ),\n"
        "    }\n\n"
        "# Examples:\n"
        "# row = bulk_rows[0]\n"
        "# arrays = load_bulk_data(row)\n"
        "# ambiguous = [row for row in bulk_rows if row['status'] == 'AMBG']\n"
        "# NVB-BULK-END\n"
    )


def write_bulk_debug_python(
    output: BulkDebugOutput | None, bulk_rows: list[dict[str, Any]]
) -> None:
    if output is None:
        return

    script = format_bulk_debug_python(bulk_rows)
    if output.is_stdout:
        print(script, end="")
        return

    with open(output.destination, "w", encoding="utf-8") as output_file:
        output_file.write(script)


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


def symmetric_frac_diff(delta_t1, delta_t2):
    return (delta_t1 - delta_t2) / min(delta_t1, delta_t2)


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


def make_timing_interval(lower, upper, center):
    if (
        not is_positive_finite(lower)
        or not is_positive_finite(upper)
        or not is_positive_finite(center)
        or lower > center
        or center > upper
    ):
        return None
    return TimingInterval(lower=lower, upper=upper, center=center)


def compute_robust_summary_interval(timing):
    if (
        is_positive_finite(timing.minimum)
        and is_positive_finite(timing.first_quartile)
        and is_positive_finite(timing.median)
        and is_positive_finite(timing.third_quartile)
        and timing.minimum <= timing.first_quartile
        and timing.first_quartile <= timing.median
        and timing.median <= timing.third_quartile
    ):
        return make_timing_interval(
            lower=timing.minimum,
            upper=timing.third_quartile,
            center=timing.median,
        )

    return None


def compute_mean_summary_interval(timing):
    if not is_positive_finite(timing.mean) or not is_nonnegative_finite(timing.stdev):
        return None

    # Keep the lower bound positive so later ratio/log-distance checks are defined.
    lower = max(timing.mean - timing.stdev, timing.mean * 0.001)
    upper = timing.mean + timing.stdev
    if is_positive_finite(timing.minimum):
        lower = max(lower, timing.minimum)
    if is_positive_finite(timing.maximum):
        upper = min(upper, timing.maximum)

    return make_timing_interval(lower=lower, upper=upper, center=timing.mean)


def compute_timing_interval(timing):
    robust_interval = compute_robust_summary_interval(timing)
    if robust_interval is not None:
        return robust_interval
    return compute_mean_summary_interval(timing)


def compute_timing_interval_from_samples(samples):
    timing_input = compute_robust_timing_input_from_samples(samples)
    if timing_input is None:
        return None
    _, interval = timing_input
    return interval


def percentile_rank(percentile: int, sample_count: int) -> int:
    clamped_percentile = min(max(percentile, 0), 100)
    # Use integer round-half-up arithmetic to match C++ std::round behavior.
    return (clamped_percentile * (sample_count - 1) + 50) // 100


def compute_robust_timing_input_from_samples(samples):
    values = positive_finite_array(samples)
    if values is None:
        return None
    sorted_values = np.sort(values)
    first_quartile = sorted_values[percentile_rank(25, len(sorted_values))]
    median = sorted_values[percentile_rank(50, len(sorted_values))]
    third_quartile = sorted_values[percentile_rank(75, len(sorted_values))]
    interval = make_timing_interval(
        lower=sorted_values[0],
        upper=third_quartile,
        center=median,
    )
    relative_dispersion = compute_relative_dispersion(
        third_quartile - first_quartile, median
    )
    if interval is None or relative_dispersion is None:
        return None
    return TimeEstimate(
        center=median, relative_dispersion=relative_dispersion
    ), interval


def make_decision(status, code, message, *, severity=0.0):
    return TimingDecision(
        status=status,
        reason=DecisionReason(code=code, message=message, severity=severity),
    )


def compare_intervals_for_clear_gap(ref_interval, cmp_interval, thresholds):
    # These ratios are equivalent to log(ref/cmp) >= log(1 + delta), but avoid
    # evaluating logarithms on every comparison.
    if cmp_interval.upper < ref_interval.lower:
        if (
            symmetric_frac_diff(ref_interval.lower, cmp_interval.upper)
            >= thresholds.clear_gap_relative
        ):
            return ComparisonStatus.FAST
    if cmp_interval.lower > ref_interval.upper:
        if (
            symmetric_frac_diff(cmp_interval.lower, ref_interval.upper)
            >= thresholds.clear_gap_relative
        ):
            return ComparisonStatus.SLOW
    return None


def compute_diff_interval(ref_interval, cmp_interval):
    return (
        cmp_interval.lower - ref_interval.upper,
        cmp_interval.upper - ref_interval.lower,
    )


def compute_frac_diff_interval(ref_interval, cmp_interval):
    # Report change using the same symmetric relative-distance family as the
    # robust clear-gap checks. Reversing reference/compare flips the sign while
    # preserving the magnitude.
    return (
        symmetric_frac_diff(cmp_interval.lower, ref_interval.upper),
        symmetric_frac_diff(cmp_interval.upper, ref_interval.lower),
    )


def centers_are_close(ref_center, cmp_center, thresholds):
    if not is_positive_finite(ref_center) or not is_positive_finite(cmp_center):
        return False
    return abs(symmetric_frac_diff(ref_center, cmp_center)) <= (
        thresholds.same_center_relative
    )


def interval_overlap_fraction(ref_interval, cmp_interval):
    intersection_lower = max(ref_interval.lower, cmp_interval.lower)
    intersection_upper = min(ref_interval.upper, cmp_interval.upper)
    if intersection_upper < intersection_lower:
        return 0.0

    ref_width = ref_interval.upper - ref_interval.lower
    cmp_width = cmp_interval.upper - cmp_interval.lower
    min_width = min(ref_width, cmp_width)
    if min_width > 0.0:
        return (intersection_upper - intersection_lower) / min_width

    if ref_width == 0.0 and cmp_width == 0.0:
        return 1.0 if ref_interval.lower == cmp_interval.lower else 0.0

    if ref_width == 0.0:
        return (
            1.0
            if cmp_interval.lower <= ref_interval.lower <= cmp_interval.upper
            else 0.0
        )

    return (
        1.0 if ref_interval.lower <= cmp_interval.lower <= ref_interval.upper else 0.0
    )


def intervals_overlap_strongly(ref_interval, cmp_interval, thresholds):
    return (
        interval_overlap_fraction(ref_interval, cmp_interval)
        >= thresholds.same_overlap_fraction
    )


def nearest_distances_to_sorted(target, source):
    pos = np.searchsorted(source, target, side="left")
    left = np.clip(pos - 1, 0, len(source) - 1)
    right = np.clip(pos, 0, len(source) - 1)
    return np.minimum(
        np.abs(target - source[left]),
        np.abs(target - source[right]),
    )


def symmetric_nearest_distances(x, y):
    # This is O(N log M + M log N), but runs in NumPy C code and operates on
    # unique supports. If this becomes a bottleneck for very large supports,
    # add an optional O(N + M) two-pass merge helper to cuda.bench and fall back
    # to this implementation when cuda.bench is unavailable.
    return nearest_distances_to_sorted(x, y), nearest_distances_to_sorted(y, x)


def symmetric_nearest_log_distances(x, y):
    return symmetric_nearest_distances(np.log(x), np.log(y))


def compute_effective_support_mask(counts, thresholds):
    """Return the unique-value mask used for support coverage.

    Sample-weight coverage always uses all values. Support coverage may ignore
    low-count values only when their total sample mass is small; otherwise it
    falls back to full support, preserving all-unique datasets.
    """
    counts = np.asarray(counts)
    total_count = np.sum(counts)
    if (
        len(counts) == 0
        or total_count <= 0
        or thresholds.bulk_support_rare_sample_fraction <= 0.0
        or thresholds.bulk_support_max_removed_sample_fraction <= 0.0
    ):
        return np.ones(len(counts), dtype=bool), SupportFilterInfo(
            activated=False,
            reason="disabled",
            removed_sample_fraction=0.0,
        )

    if np.all(counts == 1):
        return np.ones(len(counts), dtype=bool), SupportFilterInfo(
            activated=False,
            reason="all_values_unique",
            removed_sample_fraction=0.0,
        )

    min_count = max(
        2,
        math.ceil(thresholds.bulk_support_rare_sample_fraction * total_count),
    )
    support_mask = counts >= min_count
    if np.all(support_mask):
        return np.ones(len(counts), dtype=bool), SupportFilterInfo(
            activated=False,
            reason="no_rare_values",
            removed_sample_fraction=0.0,
        )
    if not np.any(support_mask):
        return np.ones(len(counts), dtype=bool), SupportFilterInfo(
            activated=False,
            reason="would_remove_all_support",
            removed_sample_fraction=0.0,
        )

    removed_sample_fraction = np.sum(counts[~support_mask]) / total_count
    if removed_sample_fraction > thresholds.bulk_support_max_removed_sample_fraction:
        return np.ones(len(counts), dtype=bool), SupportFilterInfo(
            activated=False,
            reason="would_remove_too_much_mass",
            removed_sample_fraction=0.0,
        )

    return support_mask, SupportFilterInfo(
        activated=True,
        reason="filtered",
        removed_sample_fraction=removed_sample_fraction,
    )


def format_support_filter_info(filter_info):
    if filter_info.activated:
        return f"on({format_coverage(filter_info.removed_sample_fraction)})"

    if filter_info.reason == "no_rare_values":
        return "off(no rare values)"
    if filter_info.reason == "all_values_unique":
        return "off(all values unique)"
    if filter_info.reason == "would_remove_too_much_mass":
        return "off(would remove too much mass)"
    if filter_info.reason == "would_remove_all_support":
        return "off(would remove all support)"
    return "off(disabled)"


def sorted_unique_counts(values: Float32Array) -> tuple[NumpyArray, NumpyArray]:
    unique_values, unique_counts = np.unique(values, return_counts=True)
    # unique is not guaranteed to return sorted values
    # make sure to order them
    sorting_indices = np.argsort(unique_values)
    return unique_values[sorting_indices], unique_counts[sorting_indices]


def compute_nearest_neighbor_coverages(
    ref_values: Float32Array, cmp_values: Float32Array, thresholds: ComparisonThresholds
) -> dict[str, Any] | None:
    ref_unique, ref_counts = sorted_unique_counts(ref_values)
    cmp_unique, cmp_counts = sorted_unique_counts(cmp_values)
    if len(ref_unique) == 0 or len(cmp_unique) == 0:
        return None

    ref_distances, cmp_distances = symmetric_nearest_log_distances(
        ref_unique, cmp_unique
    )
    tolerance = math.log1p(thresholds.same_center_relative)
    ref_covered = ref_distances <= tolerance
    cmp_covered = cmp_distances <= tolerance
    ref_support_mask, ref_filter_info = compute_effective_support_mask(
        ref_counts, thresholds
    )
    cmp_support_mask, cmp_filter_info = compute_effective_support_mask(
        cmp_counts, thresholds
    )

    return {
        "ref_sample": np.sum(ref_counts[ref_covered]) / np.sum(ref_counts),
        "cmp_sample": np.sum(cmp_counts[cmp_covered]) / np.sum(cmp_counts),
        "ref_support": np.mean(ref_covered[ref_support_mask]),
        "cmp_support": np.mean(cmp_covered[cmp_support_mask]),
        "ref_support_filter": ref_filter_info,
        "cmp_support_filter": cmp_filter_info,
    }


def coverages_support_same(coverages, thresholds):
    return (
        coverages["ref_sample"] >= thresholds.bulk_same_sample_coverage
        and coverages["cmp_sample"] >= thresholds.bulk_same_sample_coverage
        and coverages["ref_support"] >= thresholds.bulk_same_support_coverage
        and coverages["cmp_support"] >= thresholds.bulk_same_support_coverage
    )


def format_coverage_threshold(threshold):
    return f"{threshold * 100.0:.1f}%"


def format_coverage(value):
    return f"{value * 100.0:.1f}%"


def make_bulk_coverage_mismatch_decision(label, coverages, thresholds):
    sample_threshold = format_coverage_threshold(thresholds.bulk_same_sample_coverage)
    support_threshold = format_coverage_threshold(thresholds.bulk_same_support_coverage)
    sample_deficit = max(
        thresholds.bulk_same_sample_coverage - coverages["ref_sample"],
        thresholds.bulk_same_sample_coverage - coverages["cmp_sample"],
        0.0,
    )
    support_deficit = max(
        thresholds.bulk_same_support_coverage - coverages["ref_support"],
        thresholds.bulk_same_support_coverage - coverages["cmp_support"],
        0.0,
    )
    severity = max(sample_deficit, support_deficit)
    return make_decision(
        ComparisonStatus.UNDECIDED,
        f"bulk_{label}_support_mismatch",
        f"sample: min(ref={format_coverage(coverages['ref_sample'])}, "
        f"cmp={format_coverage(coverages['cmp_sample'])}) >= {sample_threshold}; "
        f"support: min(ref={format_coverage(coverages['ref_support'])}, "
        f"cmp={format_coverage(coverages['cmp_support'])}) >= {support_threshold}",
        severity=severity,
    )


def positive_finite_array(values):
    if values is None or len(values) == 0:
        return None

    array = np.asarray(values, dtype=np.float64)
    if np.all(np.isfinite(array) & (array > 0.0)):
        return array
    return None


def get_bulk_time_and_cycles(timing):
    samples = positive_finite_array(timing.samples)
    frequencies = positive_finite_array(timing.frequencies)
    if samples is None or frequencies is None:
        return None
    if len(samples) != len(frequencies):
        return None
    return samples, samples * frequencies


def has_material_bulk_source(source):
    if source is None:
        return False

    has_material_payload = getattr(source, "has_material_payload", None)
    if callable(has_material_payload):
        return has_material_payload()

    values = getattr(source, "values", None)
    return values is not None and len(values) > 0


def has_material_bulk_cycle_sources(timing):
    return has_material_bulk_source(timing.sample_source) and has_material_bulk_source(
        timing.frequency_source
    )


def scale_interval(interval, scale):
    if not is_positive_finite(scale):
        return None
    return make_timing_interval(
        lower=interval.lower * scale,
        upper=interval.upper * scale,
        center=interval.center * scale,
    )


def confirm_clear_gap_with_clock_rate(
    status, ref_timing, cmp_timing, ref_interval, cmp_interval, thresholds
):
    if ref_timing.sm_clock_rate_mean is None or cmp_timing.sm_clock_rate_mean is None:
        return make_decision(
            ComparisonStatus.UNDECIDED,
            "missing_clock_rate",
            "clear timing gap was not confirmed because SM clock summaries are unavailable",
        )

    ref_cycles = scale_interval(ref_interval, ref_timing.sm_clock_rate_mean)
    cmp_cycles = scale_interval(cmp_interval, cmp_timing.sm_clock_rate_mean)
    if ref_cycles is None or cmp_cycles is None:
        return make_decision(
            ComparisonStatus.UNDECIDED,
            "invalid_clock_rate",
            "clear timing gap was not confirmed because SM clock summaries are invalid",
        )

    cycle_status = compare_intervals_for_clear_gap(ref_cycles, cmp_cycles, thresholds)
    if cycle_status == status:
        return make_decision(
            status,
            "clear_gap_confirmed_by_summary_cycles",
            "clear timing gap was confirmed by SM-clock-adjusted cycle intervals",
        )
    return make_decision(
        ComparisonStatus.UNDECIDED,
        "summary_cycle_gap_not_confirmed",
        "clear timing gap was not confirmed by SM-clock-adjusted cycle intervals",
    )


def confirm_clear_gap_with_bulk_cycles(status, ref_timing, cmp_timing, thresholds):
    # Only suppress the summary-clock fallback when both inputs advertise paired,
    # non-empty bulk payloads. Missing or empty files are treated as unavailable;
    # malformed non-empty payloads become AMBG after the lazy read below.
    has_bulk_cycle_sources = has_material_bulk_cycle_sources(
        ref_timing
    ) and has_material_bulk_cycle_sources(cmp_timing)
    if not has_bulk_cycle_sources:
        return None

    ref_bulk = get_bulk_time_and_cycles(ref_timing)
    cmp_bulk = get_bulk_time_and_cycles(cmp_timing)
    if ref_bulk is None or cmp_bulk is None:
        return make_decision(
            ComparisonStatus.UNDECIDED,
            "bulk_cycle_data_unusable",
            "bulk sample time and frequency data are present but unusable for cycle confirmation",
        )

    _, ref_cycles = ref_bulk
    _, cmp_cycles = cmp_bulk
    ref_cycle_interval = compute_timing_interval_from_samples(ref_cycles)
    cmp_cycle_interval = compute_timing_interval_from_samples(cmp_cycles)
    if ref_cycle_interval is None or cmp_cycle_interval is None:
        return make_decision(
            ComparisonStatus.UNDECIDED,
            "bulk_cycle_data_unusable",
            "bulk cycle intervals could not be constructed",
        )

    cycle_status = compare_intervals_for_clear_gap(
        ref_cycle_interval, cmp_cycle_interval, thresholds
    )
    if cycle_status == status:
        return make_decision(
            status,
            "clear_gap_confirmed_by_bulk_cycles",
            "clear timing gap was confirmed by bulk cycle intervals",
        )
    return make_decision(
        ComparisonStatus.UNDECIDED,
        "bulk_cycle_gap_not_confirmed",
        "clear timing gap was not confirmed by bulk cycle intervals",
    )


def compare_timings_for_clear_gap(
    ref_timing, cmp_timing, ref_interval, cmp_interval, thresholds
):
    if ref_interval is None or cmp_interval is None:
        return make_decision(
            ComparisonStatus.UNDECIDED,
            "missing_interval",
            "could not construct comparable timing intervals",
        )

    status = compare_intervals_for_clear_gap(ref_interval, cmp_interval, thresholds)
    if status is None:
        return make_decision(
            ComparisonStatus.UNDECIDED,
            "no_clear_gap",
            "timing intervals do not have a sufficient clear gap",
        )

    bulk_decision = confirm_clear_gap_with_bulk_cycles(
        status, ref_timing, cmp_timing, thresholds
    )
    if bulk_decision is not None:
        return bulk_decision

    return confirm_clear_gap_with_clock_rate(
        status, ref_timing, cmp_timing, ref_interval, cmp_interval, thresholds
    )


def compare_intervals_for_same(ref_interval, cmp_interval, thresholds):
    if not centers_are_close(ref_interval.center, cmp_interval.center, thresholds):
        return make_decision(
            ComparisonStatus.UNDECIDED,
            "centers_not_close",
            "timing centers are not close enough to declare same",
        )
    if not intervals_overlap_strongly(ref_interval, cmp_interval, thresholds):
        return make_decision(
            ComparisonStatus.UNDECIDED,
            "weak_interval_overlap",
            "timing intervals do not overlap strongly enough to declare same",
        )
    return make_decision(
        ComparisonStatus.SAME,
        "same_summary",
        "timing centers are close and intervals overlap strongly",
    )


def confirm_same_with_clock_rate(
    ref_timing, cmp_timing, ref_interval, cmp_interval, thresholds
):
    if ref_timing.sm_clock_rate_mean is None or cmp_timing.sm_clock_rate_mean is None:
        return make_decision(
            ComparisonStatus.SAME,
            "same_without_clock_rate",
            "timing centers are close and intervals overlap strongly; SM clock summaries are unavailable",
        )

    ref_cycles = scale_interval(ref_interval, ref_timing.sm_clock_rate_mean)
    cmp_cycles = scale_interval(cmp_interval, cmp_timing.sm_clock_rate_mean)
    if ref_cycles is None or cmp_cycles is None:
        return make_decision(
            ComparisonStatus.UNDECIDED,
            "invalid_clock_rate",
            "same decision was not confirmed because SM clock summaries are invalid",
        )

    decision = compare_intervals_for_same(ref_cycles, cmp_cycles, thresholds)
    if decision.status == ComparisonStatus.SAME:
        return make_decision(
            ComparisonStatus.SAME,
            "same_confirmed_by_cycles",
            "timing and SM-clock-adjusted cycle intervals both support same",
        )
    return make_decision(
        ComparisonStatus.UNDECIDED,
        "cycle_same_not_confirmed",
        "same decision was not confirmed by SM-clock-adjusted cycle intervals",
    )


def compare_values_for_bulk_same(ref_values, cmp_values, *, label, thresholds):
    coverages = compute_nearest_neighbor_coverages(ref_values, cmp_values, thresholds)
    if coverages is None:
        return make_decision(
            ComparisonStatus.UNDECIDED,
            f"bulk_{label}_data_unusable",
            f"bulk {label} data is empty or unusable",
        )
    if coverages_support_same(coverages, thresholds):
        return make_decision(
            ComparisonStatus.SAME,
            f"bulk_{label}_same",
            f"bulk {label} nearest-neighbor coverage supports same",
        )
    return make_bulk_coverage_mismatch_decision(label, coverages, thresholds)


def compare_timings_for_bulk_same(ref_timing, cmp_timing, thresholds):
    ref_times = positive_finite_array(ref_timing.samples)
    cmp_times = positive_finite_array(cmp_timing.samples)
    if ref_times is None or cmp_times is None:
        if (
            ref_timing.sample_source is not None
            and cmp_timing.sample_source is not None
        ):
            return make_decision(
                ComparisonStatus.UNDECIDED,
                "bulk_time_data_unusable",
                "bulk time data is empty or unusable",
            )
        return make_decision(
            ComparisonStatus.UNDECIDED,
            "bulk_data_unavailable",
            "bulk sample time data are unavailable",
        )

    time_decision = compare_values_for_bulk_same(
        ref_times, cmp_times, label="time", thresholds=thresholds
    )
    if time_decision.status != ComparisonStatus.SAME:
        return time_decision

    if ref_timing.frequency_source is None or cmp_timing.frequency_source is None:
        return make_decision(
            ComparisonStatus.SAME,
            "bulk_time_same_without_cycles",
            "bulk time nearest-neighbor coverage supports same; sample frequencies are unavailable",
        )

    ref_frequencies = positive_finite_array(ref_timing.frequencies)
    cmp_frequencies = positive_finite_array(cmp_timing.frequencies)
    if (
        ref_frequencies is None
        or cmp_frequencies is None
        or len(ref_times) != len(ref_frequencies)
        or len(cmp_times) != len(cmp_frequencies)
    ):
        return make_decision(
            ComparisonStatus.UNDECIDED,
            "bulk_cycle_data_unusable",
            "bulk cycle data is empty or unusable",
        )

    ref_cycles = ref_times * ref_frequencies
    cmp_cycles = cmp_times * cmp_frequencies
    cycle_decision = compare_values_for_bulk_same(
        ref_cycles, cmp_cycles, label="cycle", thresholds=thresholds
    )
    if cycle_decision.status != ComparisonStatus.SAME:
        return cycle_decision

    return make_decision(
        ComparisonStatus.SAME,
        "bulk_same",
        "bulk time and cycle nearest-neighbor coverage both support same",
    )


def compare_timings_for_bulk_same_if_available(ref_timing, cmp_timing, thresholds):
    decision = compare_timings_for_bulk_same(ref_timing, cmp_timing, thresholds)
    if decision.reason.code == "bulk_data_unavailable":
        return None
    return decision


def confirm_bulk_time_same_without_cycles(
    ref_timing, cmp_timing, ref_interval, cmp_interval, thresholds
):
    decision = confirm_same_with_clock_rate(
        ref_timing, cmp_timing, ref_interval, cmp_interval, thresholds
    )
    if decision.reason.code == "same_without_clock_rate":
        return make_decision(
            ComparisonStatus.SAME,
            "bulk_time_same_without_cycles",
            "bulk time nearest-neighbor coverage supports same; cycle data are unavailable",
        )
    if decision.status == ComparisonStatus.SAME:
        return make_decision(
            ComparisonStatus.SAME,
            "bulk_time_same_confirmed_by_summary_cycles",
            "bulk time nearest-neighbor coverage supports same and SM-clock-adjusted summary intervals confirm same",
        )
    return decision


def compare_timings_for_same(
    ref_timing, cmp_timing, ref_noise, cmp_noise, ref_interval, cmp_interval, thresholds
):
    if not is_usable_noise(ref_noise) or not is_usable_noise(cmp_noise):
        return make_decision(
            ComparisonStatus.UNDECIDED,
            "noise_unavailable",
            "relative dispersion is unavailable, negative, or non-finite",
        )

    if ref_interval is None or cmp_interval is None:
        return make_decision(
            ComparisonStatus.UNDECIDED,
            "missing_interval",
            "could not construct comparable timing intervals",
        )

    decision = compare_intervals_for_same(ref_interval, cmp_interval, thresholds)
    if decision.status != ComparisonStatus.SAME:
        return decision

    bulk_decision = compare_timings_for_bulk_same_if_available(
        ref_timing, cmp_timing, thresholds
    )
    if (
        bulk_decision is not None
        and bulk_decision.reason.code == "bulk_time_same_without_cycles"
    ):
        bulk_decision = confirm_bulk_time_same_without_cycles(
            ref_timing, cmp_timing, ref_interval, cmp_interval, thresholds
        )

    if max(ref_noise, cmp_noise) > thresholds.same_relative_dispersion_ceiling:
        if bulk_decision is not None:
            return bulk_decision
        return make_decision(
            ComparisonStatus.UNDECIDED,
            "noise_too_high",
            "relative dispersion is too high to declare same",
        )

    if bulk_decision is not None:
        return bulk_decision

    return confirm_same_with_clock_rate(
        ref_timing, cmp_timing, ref_interval, cmp_interval, thresholds
    )


def has_robust_estimate(summary):
    return summary.median is not None and (
        summary.interquartile_range_relative is not None
        or summary.interquartile_range is not None
    )


def has_robust_interval(summary):
    return (
        summary.minimum is not None
        and summary.first_quartile is not None
        and summary.median is not None
        and summary.third_quartile is not None
    )


def has_mean_estimate(summary):
    return summary.mean is not None and (
        summary.stdev_relative is not None or summary.stdev is not None
    )


def select_relative_dispersion(relative_dispersion, absolute_dispersion, center):
    if relative_dispersion is not None:
        return relative_dispersion
    return compute_relative_dispersion(absolute_dispersion, center)


def compute_robust_summary_estimate(timing):
    if not has_robust_estimate(timing):
        return None
    return TimeEstimate(
        center=timing.median,
        relative_dispersion=select_relative_dispersion(
            timing.interquartile_range_relative,
            timing.interquartile_range,
            timing.median,
        ),
    )


def compute_mean_summary_estimate(timing):
    if not has_mean_estimate(timing):
        return None
    return TimeEstimate(
        center=timing.mean,
        relative_dispersion=select_relative_dispersion(
            timing.stdev_relative, timing.stdev, timing.mean
        ),
    )


def compute_robust_summary_timing_input(timing):
    estimate = compute_robust_summary_estimate(timing)
    interval = compute_robust_summary_interval(timing)
    if estimate is None or interval is None:
        return None
    return estimate, interval


def compute_mean_summary_timing_input(timing):
    estimate = compute_mean_summary_estimate(timing)
    interval = compute_mean_summary_interval(timing)
    if estimate is None or interval is None:
        return None
    return estimate, interval


def make_timing_comparison_inputs(ref_input, cmp_input):
    ref_estimate, ref_interval = ref_input
    cmp_estimate, cmp_interval = cmp_input
    return TimingComparisonInputs(
        ref_estimate=ref_estimate,
        cmp_estimate=cmp_estimate,
        ref_interval=ref_interval,
        cmp_interval=cmp_interval,
    )


def compute_robust_timing_input(timing):
    robust_input = compute_robust_summary_timing_input(timing)
    if robust_input is not None:
        return robust_input
    return compute_robust_timing_input_from_samples(timing.samples)


def compute_common_time_estimates(ref_timing, cmp_timing):
    ref_robust_estimate = compute_robust_summary_estimate(ref_timing)
    cmp_robust_estimate = compute_robust_summary_estimate(cmp_timing)
    if ref_robust_estimate is not None and cmp_robust_estimate is not None:
        return ref_robust_estimate, cmp_robust_estimate

    ref_mean_estimate = compute_mean_summary_estimate(ref_timing)
    cmp_mean_estimate = compute_mean_summary_estimate(cmp_timing)
    if ref_mean_estimate is not None and cmp_mean_estimate is not None:
        return ref_mean_estimate, cmp_mean_estimate

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


def compute_timing_comparison_inputs(ref_timing, cmp_timing):
    ref_robust_input = compute_robust_timing_input(ref_timing)
    cmp_robust_input = compute_robust_timing_input(cmp_timing)
    if ref_robust_input is not None and cmp_robust_input is not None:
        return make_timing_comparison_inputs(ref_robust_input, cmp_robust_input)

    ref_mean_input = compute_mean_summary_timing_input(ref_timing)
    cmp_mean_input = compute_mean_summary_timing_input(cmp_timing)
    if ref_mean_input is not None and cmp_mean_input is not None:
        return make_timing_comparison_inputs(ref_mean_input, cmp_mean_input)

    ref_estimate, cmp_estimate = compute_common_time_estimates(ref_timing, cmp_timing)
    return TimingComparisonInputs(
        ref_estimate=ref_estimate,
        cmp_estimate=cmp_estimate,
        ref_interval=None,
        cmp_interval=None,
    )


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
        max_noise=None,
        status=decision.status,
        reason=decision.reason,
    )


def compare_gpu_timings(ref_timing, cmp_timing, comparison_thresholds=None):
    if comparison_thresholds is None:
        comparison_thresholds = get_default_thresholds()

    timing_inputs = compute_timing_comparison_inputs(ref_timing, cmp_timing)
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

    ref_interval = timing_inputs.ref_interval
    cmp_interval = timing_inputs.cmp_interval
    diff = cmp_time - ref_time
    frac_diff = diff / ref_time
    diff_interval = None
    frac_diff_interval = None
    if ref_interval is not None and cmp_interval is not None:
        diff_interval = compute_diff_interval(ref_interval, cmp_interval)
        frac_diff_interval = compute_frac_diff_interval(ref_interval, cmp_interval)

    if not is_usable_noise(ref_noise) or not is_usable_noise(cmp_noise):
        max_noise = None
    else:
        max_noise = max(ref_noise, cmp_noise)

    decision = compare_timings_for_clear_gap(
        ref_timing, cmp_timing, ref_interval, cmp_interval, comparison_thresholds
    )
    if decision.status == ComparisonStatus.UNDECIDED and decision.reason.code in {
        "no_clear_gap",
        "missing_interval",
    }:
        decision = compare_timings_for_same(
            ref_timing,
            cmp_timing,
            ref_noise,
            cmp_noise,
            ref_interval,
            cmp_interval,
            comparison_thresholds,
        )

    return SummaryComparison(
        ref_interval=ref_interval,
        cmp_interval=cmp_interval,
        ref_estimate=ref_estimate,
        cmp_estimate=cmp_estimate,
        ref_time=ref_time,
        cmp_time=cmp_time,
        ref_noise=ref_noise,
        cmp_noise=cmp_noise,
        diff=diff,
        frac_diff=frac_diff,
        diff_interval=diff_interval,
        frac_diff_interval=frac_diff_interval,
        max_noise=max_noise,
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


def format_timing_with_interval(
    center, interval, *, center_width=None, interval_width=None
):
    if center is None or not is_positive_finite(center):
        return "n/a"
    if interval is None:
        if center_width is not None and interval_width is not None:
            center_multiplier, center_units = select_duration_units(center)
            center_text = f"{center * center_multiplier:0.3f}"
            center_text = f"{center_text:>{center_width}}"
            if interval_width == 0:
                return f"{center_text} {center_units}"
            return f"{center_text} {' ' * interval_width} {center_units}"
        return format_duration(center)

    lower_delta = interval.lower - interval.center
    upper_delta = interval.upper - interval.center
    center_multiplier, center_units = select_duration_units(center)
    delta_multiplier, delta_units = select_duration_units(lower_delta, upper_delta)
    precision = duration_precision_for_center(center, delta_multiplier)
    if center_units == delta_units:
        center_text = f"{center * center_multiplier:0.3f}"
        interval_text = (
            f"[{lower_delta * delta_multiplier:+0.{precision}f}, "
            f"{upper_delta * delta_multiplier:+0.{precision}f}]"
        )
        if center_width is not None:
            center_text = f"{center_text:>{center_width}}"
        if interval_width is not None:
            interval_text = f"{interval_text:>{interval_width}}"
        return f"{center_text} {interval_text} {center_units}"

    return (
        f"{format_duration(center)} "
        f"[{lower_delta * delta_multiplier:+0.{precision}f}, "
        f"{upper_delta * delta_multiplier:+0.{precision}f}] {delta_units}"
    )


def longest_common_prefix(strings):
    if not strings:
        return ""
    prefix = strings[0]
    for text in strings[1:]:
        while not text.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix


def common_numeric_prefix_is_useful(prefix):
    if "." not in prefix:
        return False

    numeric_digits = sum(char.isdigit() for char in prefix)
    fractional_prefix = prefix.split(".", 1)[1]
    fractional_digits = sum(char.isdigit() for char in fractional_prefix)
    return numeric_digits >= 2 and fractional_digits >= 1


def align_interval_values(values, widths=None):
    if widths is None:
        widths = [max(len(value) for value in values)] * len(values)
    return [f"{value:>{width}}" for value, width in zip(values, widths, strict=True)]


def explicit_interval_values(center, interval):
    if (
        not is_positive_finite(interval.lower)
        or not is_positive_finite(interval.center)
        or not is_positive_finite(interval.upper)
    ):
        return None

    multiplier, units = select_duration_units(
        interval.lower, interval.center, interval.upper
    )
    return (
        [
            f"{interval.lower * multiplier:0.3f}",
            f"{interval.center * multiplier:0.3f}",
            f"{interval.upper * multiplier:0.3f}",
        ],
        units,
    )


def explicit_interval_column_widths(comparisons, center_getter, interval_getter):
    widths = [0, 0, 0]
    for comparison in comparisons:
        center = center_getter(comparison)
        interval = interval_getter(comparison)
        if not is_positive_finite(center) or interval is None:
            continue

        interval_values = explicit_interval_values(center, interval)
        if interval_values is None:
            continue
        values, _ = interval_values
        prefix = longest_common_prefix(values)
        if common_numeric_prefix_is_useful(prefix):
            continue

        widths = [max(width, len(value)) for width, value in zip(widths, values)]
    return widths


def format_timing_with_explicit_interval(center, interval, *, value_widths=None):
    if center is None or not is_positive_finite(center):
        return "n/a"
    if interval is None:
        return format_duration(center)

    interval_values = explicit_interval_values(center, interval)
    if interval_values is None:
        return "n/a"
    values, units = interval_values
    prefix = longest_common_prefix(values)
    if not common_numeric_prefix_is_useful(prefix):
        values = align_interval_values(values, value_widths)
        return f"[{values[0]} | {values[1]} | {values[2]}] {units}"

    suffixes = [value[len(prefix) :] for value in values]
    return f"{prefix}[{suffixes[0]} | {suffixes[1]} | {suffixes[2]}] {units}"


def format_percentage(percentage):
    if percentage is None:
        return "n/a"
    if math.isnan(percentage):
        return "n/a"
    if math.isinf(percentage):
        return "inf"
    return f"{percentage * 100.0:0.2f}%"


def format_percentage_bounds(bounds, status):
    if bounds is None:
        return "n/a"
    lower, upper = bounds
    if status == ComparisonStatus.FAST:
        return f"<= {upper * 100.0:+0.1f}%"
    if status == ComparisonStatus.SLOW:
        return f">= {lower * 100.0:+0.1f}%"
    return f"in [{lower * 100.0:+0.1f}%, {upper * 100.0:+0.1f}%]"


def format_change(comparison):
    if comparison.status not in {ComparisonStatus.FAST, ComparisonStatus.SLOW}:
        return ""
    return format_percentage_bounds(comparison.frac_diff_interval, comparison.status)


def format_center_diff(ref_time, cmp_time):
    if not is_positive_finite(ref_time) or not is_positive_finite(cmp_time):
        return "n/a"
    return f"{symmetric_frac_diff(cmp_time, ref_time) * 100.0:+0.1f}%"


def format_interval_span(interval):
    if interval is None or not is_positive_finite(interval.center):
        return "n/a"
    if not is_finite(interval.lower) or not is_finite(interval.upper):
        return "n/a"
    if interval.upper < interval.lower:
        return "n/a"
    return f"{((interval.upper - interval.lower) / interval.center) * 100.0:0.1f}%"


def get_display_headers(display):
    if display == "explain":
        return (
            [
                "Ref [Lo | Ce | Hi]",
                "Cmp [Lo | Ce | Hi]",
                "Ref Noise",
                "Cmp Noise",
                "Reason",
                "Change",
                "Status",
            ],
            ["right", "right", "right", "right", "left", "right", "center"],
        )
    if display == "simple":
        return (
            ["Ref", "Ref Span", "Cmp", "Cmp Span", "%C Diff", "Change", "Status"],
            ["right", "right", "right", "right", "right", "right", "center"],
        )
    return (
        ["Ref", "Cmp", "Change", "Status"],
        ["right", "right", "right", "center"],
    )


def append_display_row(row, comparison, no_color, display):
    if display == "simple":
        row.append(format_duration(comparison.ref_time))
        row.append(format_interval_span(comparison.ref_interval))
        row.append(format_duration(comparison.cmp_time))
        row.append(format_interval_span(comparison.cmp_interval))
        row.append(format_center_diff(comparison.ref_time, comparison.cmp_time))
        row.append(format_change(comparison))
        row.append(colorize_comparison_status(comparison.status, no_color))
        return

    row.append(
        format_timing_with_interval(comparison.ref_time, comparison.ref_interval)
    )
    row.append(
        format_timing_with_interval(comparison.cmp_time, comparison.cmp_interval)
    )
    if display == "explain":
        row[-2] = format_timing_with_explicit_interval(
            comparison.ref_time, comparison.ref_interval
        )
        row[-1] = format_timing_with_explicit_interval(
            comparison.cmp_time, comparison.cmp_interval
        )
        row.append(format_percentage(comparison.ref_noise))
        row.append(format_percentage(comparison.cmp_noise))
        row.append(format_reason_display_code(comparison.reason.code))
    row.append(format_change(comparison))
    row.append(colorize_comparison_status(comparison.status, no_color))


def align_explain_interval_columns(rows, comparisons, axis_count):
    ref_widths = explicit_interval_column_widths(
        comparisons,
        lambda comparison: comparison.ref_time,
        lambda comparison: comparison.ref_interval,
    )
    cmp_widths = explicit_interval_column_widths(
        comparisons,
        lambda comparison: comparison.cmp_time,
        lambda comparison: comparison.cmp_interval,
    )
    for row, comparison in zip(rows, comparisons, strict=True):
        row[axis_count] = format_timing_with_explicit_interval(
            comparison.ref_time, comparison.ref_interval, value_widths=ref_widths
        )
        row[axis_count + 1] = format_timing_with_explicit_interval(
            comparison.cmp_time, comparison.cmp_interval, value_widths=cmp_widths
        )


def timing_interval_column_widths(comparisons, center_getter, interval_getter):
    center_width = 0
    interval_width = 0
    for comparison in comparisons:
        center = center_getter(comparison)
        if not is_positive_finite(center):
            continue

        center_multiplier, center_units = select_duration_units(center)
        center_text = f"{center * center_multiplier:0.3f}"
        center_width = max(center_width, len(center_text))

        interval = interval_getter(comparison)
        if interval is None:
            continue

        lower_delta = interval.lower - interval.center
        upper_delta = interval.upper - interval.center
        delta_multiplier, delta_units = select_duration_units(lower_delta, upper_delta)
        if center_units != delta_units:
            continue

        precision = duration_precision_for_center(center, delta_multiplier)
        interval_text = (
            f"[{lower_delta * delta_multiplier:+0.{precision}f}, "
            f"{upper_delta * delta_multiplier:+0.{precision}f}]"
        )
        interval_width = max(interval_width, len(interval_text))

    return center_width, interval_width


def align_timing_interval_columns(rows, comparisons, axis_count):
    ref_center_width, ref_interval_width = timing_interval_column_widths(
        comparisons,
        lambda comparison: comparison.ref_time,
        lambda comparison: comparison.ref_interval,
    )
    cmp_center_width, cmp_interval_width = timing_interval_column_widths(
        comparisons,
        lambda comparison: comparison.cmp_time,
        lambda comparison: comparison.cmp_interval,
    )
    for row, comparison in zip(rows, comparisons, strict=True):
        row[axis_count] = format_timing_with_interval(
            comparison.ref_time,
            comparison.ref_interval,
            center_width=ref_center_width,
            interval_width=ref_interval_width,
        )
        row[axis_count + 1] = format_timing_with_interval(
            comparison.cmp_time,
            comparison.cmp_interval,
            center_width=cmp_center_width,
            interval_width=cmp_interval_width,
        )


def is_usable_noise(noise):
    return is_nonnegative_finite(noise)


def colorize_comparison_status(status, no_color):
    if status == ComparisonStatus.UNKNOWN:
        fore_name = "YELLOW"
        emoji = Emoji.YELLOW
    elif status == ComparisonStatus.UNDECIDED:
        fore_name = "LIGHTBLACK_EX"
        emoji = Emoji.SHRUG
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
        tool_name=current_tool_name(),
    )
    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")

    plt = require_tooling_dependency(
        ToolingDependency(
            "matplotlib.pyplot", "matplotlib", "plot rendering", extra="plot"
        ),
        tool_name=current_tool_name(),
    )
    ticker = require_tooling_dependency(
        ToolingDependency(
            "matplotlib.ticker", "matplotlib", "plot axis formatting", extra="plot"
        ),
        tool_name=current_tool_name(),
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
        output = "nvbench_compare_robust.png"
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
    ref_json_path=None,
    cmp_json_path=None,
    comparison_thresholds=None,
    display="intervals",
    bulk_debug_rows=None,
):
    if comparison_thresholds is None:
        comparison_thresholds = get_default_thresholds()

    if plot_along:
        plt = require_tooling_dependency(
            ToolingDependency(
                "matplotlib.pyplot",
                "matplotlib",
                "per-axis plot rendering",
                extra="plot",
            ),
            tool_name=current_tool_name(),
        )
        sns = require_tooling_dependency(
            ToolingDependency(
                "seaborn", "seaborn", "per-axis plot styling", extra="plot"
            ),
            tool_name=current_tool_name(),
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
                "nvbench-compare-robust pairs devices by position, so each compared "
                "benchmark must contain the same number of devices"
            )

        print(f"""# {cmp_bench["name"]}\n""")

        axes = cmp_bench["axes"]
        ref_states = ref_bench["states"]
        cmp_states = cmp_bench["states"]

        axes = axes if axes else []

        headers = [x["name"] for x in axes]
        colalign = ["center"] * len(headers)
        display_headers, display_colalign = get_display_headers(display)
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
            row_comparisons = []
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
                if missing_summaries_decision is None:
                    cmp_gpu_time = extract_gpu_timing_data(
                        cmp_summaries, cmp_json_dir, json_path=cmp_json_path
                    )
                    ref_gpu_time = extract_gpu_timing_data(
                        ref_summaries, ref_json_dir, json_path=ref_json_path
                    )
                    comparison = compare_gpu_timings(
                        ref_gpu_time, cmp_gpu_time, comparison_thresholds
                    )
                else:
                    cmp_gpu_time = (
                        extract_gpu_timing_data(
                            cmp_summaries, cmp_json_dir, json_path=cmp_json_path
                        )
                        if cmp_summaries
                        else make_empty_gpu_timing_data()
                    )
                    ref_gpu_time = (
                        extract_gpu_timing_data(
                            ref_summaries, ref_json_dir, json_path=ref_json_path
                        )
                        if ref_summaries
                        else make_empty_gpu_timing_data()
                    )
                    timing_inputs = compute_timing_comparison_inputs(
                        ref_gpu_time, cmp_gpu_time
                    )
                    comparison = make_unavailable_timing_comparison(
                        missing_summaries_decision, timing_inputs
                    )
                if comparison is None:
                    continue

                if (
                    plot_along
                    and is_positive_finite(comparison.ref_time)
                    and is_positive_finite(comparison.cmp_time)
                ):
                    axis_value, axis_name_parts = extract_plot_axis_value(
                        axis_values, plot_along, cmp_bench["name"], cmp_state_name
                    )
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
                    plot_data["cmp_noise"][axis_name][axis_value] = comparison.cmp_noise
                    plot_data["ref_noise"][axis_name][axis_value] = comparison.ref_noise

                run_data.stats.record(comparison.status, comparison.reason)
                if comparison.status == ComparisonStatus.UNKNOWN or (
                    comparison.frac_diff is not None
                    and abs(comparison.frac_diff) >= threshold
                ):
                    axis_filters = matching_axis_filters(cmp_state, axis_filter_groups)
                    append_display_row(row, comparison, no_color, display)

                    rows.append(row)
                    row_comparisons.append(comparison)
                    if bulk_debug_rows is not None:
                        bulk_debug_rows.append(
                            make_bulk_debug_row(
                                row_index=len(bulk_debug_rows),
                                table_row_index=len(rows) - 1,
                                benchmark_name=cmp_bench["name"],
                                ref_json_path=ref_json_path,
                                cmp_json_path=cmp_json_path,
                                ref_device_id=ref_device_id,
                                cmp_device_id=cmp_device_id,
                                cmp_state_name=cmp_state_name,
                                occurrence=occurrence,
                                occurrence_count=cmp_state_counts[cmp_state_key],
                                axis_values=axis_values,
                                axes=axes,
                                ref_timing=ref_gpu_time,
                                cmp_timing=cmp_gpu_time,
                                comparison=comparison,
                            )
                        )
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

            has_rows = len(rows) > 0
            has_plot_along_data = bool(plot_along) and any(
                axis_times for axis_times in plot_data["cmp"].values()
            )

            cmp_device = find_device_by_id(cmp_device_id, run_data.cmp_devices)
            ref_device = find_device_by_id(ref_device_id, run_data.ref_devices)
            if ref_device is None or cmp_device is None:
                raise ValueError(
                    f"benchmark {cmp_bench['name']!r} references device pair "
                    f"ref={ref_device_id} cmp={cmp_device_id}, but device metadata is missing"
                )

            if not has_rows and not has_plot_along_data:
                continue

            if has_rows:
                if display == "explain":
                    align_explain_interval_columns(rows, row_comparisons, len(axes))
                elif display == "intervals":
                    align_timing_interval_columns(rows, row_comparisons, len(axes))

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

            if has_plot_along_data:
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
        "--preset",
        choices=sorted(COMPARISON_THRESHOLD_PRESETS),
        default=None,
        help="comparison threshold preset",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="comparison threshold TOML config",
    )
    parser.add_argument(
        "--dump-config",
        action="store_true",
        help="print the effective comparison threshold config and exit",
    )
    parser.add_argument(
        "--display",
        choices=["simple", "intervals", "explain"],
        default="intervals",
        help="comparison table display mode",
    )
    parser.add_argument(
        "--bulk-debug-python",
        default=None,
        help=(
            "Write Python code that describes bulk sample/frequency files for "
            "each displayed row. Use 'stdout' to print the code to stdout."
        ),
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
        validate_threshold_diff(args.threshold)
    except ValueError as exc:
        print(str(exc))
        return 1

    try:
        comparison_preset, comparison_thresholds = resolve_comparison_thresholds(
            args.preset, args.config
        )
    except ValueError as exc:
        print(str(exc))
        return 1

    if args.dump_config:
        print(dump_comparison_config(comparison_preset, comparison_thresholds), end="")
        return 0

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

    bulk_debug_output = (
        None
        if args.bulk_debug_python is None
        else BulkDebugOutput(args.bulk_debug_python)
    )
    bulk_debug_rows: list[dict[str, Any]] | None = (
        [] if bulk_debug_output is not None else None
    )

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
                ref_json_dir=os.path.dirname(ref),
                cmp_json_dir=os.path.dirname(comp),
                ref_json_path=ref,
                cmp_json_path=comp,
                comparison_thresholds=comparison_thresholds,
                display=args.display,
                bulk_debug_rows=bulk_debug_rows,
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
    print(f"  - Unchanged   (classified as SAME): {stats.pass_count}")
    print(f"  - Improvement (clear timing gap, %Diff < 0): {stats.improvement_count}")
    print(f"  - Regression  (clear timing gap, %Diff > 0): {stats.regression_count}")
    print(f"  - Ambiguous (comparison requires more evidence): {stats.undecided_count}")
    if stats.undecided_reasons:
        print("    - Reasons:")
        for code, reason_summary in sorted(
            stats.undecided_reasons.items(),
            key=lambda item: item[1].count,
            reverse=True,
        ):
            print(f"      - {code}: {reason_summary.count} ({reason_summary.message})")
    if args.display == "explain" and stats.reason_legend:
        legend_entries = format_reason_legend_entries(stats.reason_legend)
        if legend_entries:
            print(f"  - Reason legend: {'; '.join(legend_entries)}")
    print(
        f"  - Unknown     (input data unavailable or unusable): {stats.unknown_count}"
    )
    try:
        write_bulk_debug_python(bulk_debug_output, bulk_debug_rows or [])
    except OSError as exc:
        print(f"failed to write bulk debug Python output: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
