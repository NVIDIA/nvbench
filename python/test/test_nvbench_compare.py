# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib.util
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def nvbench_compare(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "jsondiff", types.SimpleNamespace(diff=lambda *args, **kwargs: {})
    )
    monkeypatch.setitem(
        sys.modules,
        "tabulate",
        types.SimpleNamespace(
            __version__="0.8.10", tabulate=lambda *args, **kwargs: ""
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "colorama",
        types.SimpleNamespace(
            Fore=types.SimpleNamespace(
                BLUE="",
                GREEN="",
                RED="",
                RESET="",
                YELLOW="",
            )
        ),
    )

    module_path = Path(__file__).resolve().parents[1] / "scripts" / "nvbench_compare.py"
    spec = importlib.util.spec_from_file_location("nvbench_compare", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_state(nvbench_compare, name, *, mean="1.0", noise="0.01", axis_value=None):
    return {
        "name": name,
        "device": 0,
        "axis_values": []
        if axis_value is None
        else [{"name": "A", "type": "int64", "value": axis_value}],
        "summaries": [
            {
                "tag": nvbench_compare.GPU_TIME_MEAN_TAG,
                "data": [{"name": "value", "type": "float64", "value": mean}],
            },
            {
                "tag": nvbench_compare.GPU_TIME_STDEV_RELATIVE_TAG,
                "data": [{"name": "value", "type": "float64", "value": noise}],
            },
        ],
    }


def make_benchmark(states):
    return {
        "name": "bench",
        "devices": [0],
        "axes": [{"name": "A", "type": "int64", "flags": ""}]
        if any(state["axis_values"] for state in states)
        else [],
        "states": states,
    }


def set_test_devices(nvbench_compare):
    devices = [{"id": 0, "name": "Test GPU"}]
    nvbench_compare.all_ref_devices = devices
    nvbench_compare.all_cmp_devices = devices


def test_compare_benches_accepts_matching_duplicate_state_counts(nvbench_compare):
    set_test_devices(nvbench_compare)

    ref_benches = [
        make_benchmark(
            [
                make_state(nvbench_compare, "state1"),
                make_state(nvbench_compare, "state1"),
                make_state(nvbench_compare, "state2"),
            ]
        )
    ]
    cmp_benches = [
        make_benchmark(
            [
                make_state(nvbench_compare, "state1", mean="1.005"),
                make_state(nvbench_compare, "state1", mean="1.005"),
                make_state(nvbench_compare, "state2", mean="1.005"),
            ]
        )
    ]

    nvbench_compare.compare_benches(
        ref_benches,
        cmp_benches,
        threshold=0.0,
        plot_along=None,
        plot=False,
        dark=False,
        axis_filters=[],
        benchmark_filters=[],
        no_color=True,
    )

    assert nvbench_compare.config_count == 3
    assert nvbench_compare.pass_count == 3
    assert nvbench_compare.improvement_count == 0
    assert nvbench_compare.regression_count == 0
    assert nvbench_compare.unknown_count == 0


def test_compare_benches_rejects_swapped_duplicate_state_counts(nvbench_compare):
    set_test_devices(nvbench_compare)

    ref_benches = [
        make_benchmark(
            [
                make_state(nvbench_compare, "state1"),
                make_state(nvbench_compare, "state1"),
                make_state(nvbench_compare, "state1"),
                make_state(nvbench_compare, "state2"),
                make_state(nvbench_compare, "state2"),
            ]
        )
    ]
    cmp_benches = [
        make_benchmark(
            [
                make_state(nvbench_compare, "state1"),
                make_state(nvbench_compare, "state1"),
                make_state(nvbench_compare, "state2"),
                make_state(nvbench_compare, "state2"),
                make_state(nvbench_compare, "state2"),
            ]
        )
    ]

    with pytest.raises(ValueError, match="mismatched state occurrences"):
        nvbench_compare.compare_benches(
            ref_benches,
            cmp_benches,
            threshold=0.0,
            plot_along=None,
            plot=False,
            dark=False,
            axis_filters=[],
            benchmark_filters=[],
            no_color=True,
        )


def test_compare_benches_matches_duplicate_states_after_axis_filter(nvbench_compare):
    set_test_devices(nvbench_compare)

    ref_benches = [
        make_benchmark(
            [
                make_state(nvbench_compare, "state", mean="1.0", axis_value=1),
                make_state(nvbench_compare, "state", mean="2.0", axis_value=2),
            ]
        )
    ]
    cmp_benches = [
        make_benchmark(
            [
                make_state(nvbench_compare, "state", mean="2.0", axis_value=2),
                make_state(nvbench_compare, "state", mean="1.0", axis_value=1),
            ]
        )
    ]

    nvbench_compare.compare_benches(
        ref_benches,
        cmp_benches,
        threshold=0.0,
        plot_along=None,
        plot=False,
        dark=False,
        axis_filters=[{"name": "A", "values": ["2"], "display": "A=2"}],
        benchmark_filters=[],
        no_color=True,
    )

    assert nvbench_compare.config_count == 1
    assert nvbench_compare.pass_count == 1
    assert nvbench_compare.improvement_count == 0
    assert nvbench_compare.regression_count == 0
    assert nvbench_compare.unknown_count == 0
