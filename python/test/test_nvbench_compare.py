# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib.util
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def nvbench_compare(monkeypatch):
    class DummyLine:
        def get_color(self):
            return "black"

    pyplot = types.ModuleType("matplotlib.pyplot")
    pyplot.figure = lambda *args, **kwargs: None
    pyplot.xscale = lambda *args, **kwargs: None
    pyplot.yscale = lambda *args, **kwargs: None
    pyplot.xlabel = lambda *args, **kwargs: None
    pyplot.ylabel = lambda *args, **kwargs: None
    pyplot.title = lambda *args, **kwargs: None
    pyplot.plot = lambda *args, **kwargs: [DummyLine()]
    pyplot.fill_between = lambda *args, **kwargs: None
    pyplot.legend = lambda *args, **kwargs: None
    pyplot.show = lambda *args, **kwargs: None
    pyplot.close = lambda *args, **kwargs: None

    matplotlib = types.ModuleType("matplotlib")
    matplotlib.pyplot = pyplot
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)
    monkeypatch.setitem(
        sys.modules,
        "seaborn",
        types.SimpleNamespace(set_theme=lambda *args, **kwargs: None),
    )
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


def make_state(
    nvbench_compare, name, *, mean="1.0", noise="0.01", axis_value=None, device=0
):
    return {
        "name": name,
        "device": device,
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


def make_summary(nvbench_compare, tag, value):
    return {
        "tag": getattr(nvbench_compare, tag),
        "data": [{"name": "value", "type": "float64", "value": value}],
    }


def make_benchmark(states, *, name="bench"):
    devices = []
    for state in states:
        if state["device"] not in devices:
            devices.append(state["device"])

    return {
        "name": name,
        "devices": devices,
        "axes": [{"name": "A", "type": "int64", "flags": ""}]
        if any(state["axis_values"] for state in states)
        else [],
        "states": states,
    }


def set_test_devices(monkeypatch, nvbench_compare):
    devices = [{"id": 0, "name": "Test GPU"}]
    monkeypatch.setattr(nvbench_compare, "all_ref_devices", devices)
    monkeypatch.setattr(nvbench_compare, "all_cmp_devices", devices)
    monkeypatch.setattr(nvbench_compare, "config_count", 0)
    monkeypatch.setattr(nvbench_compare, "pass_count", 0)
    monkeypatch.setattr(nvbench_compare, "improvement_count", 0)
    monkeypatch.setattr(nvbench_compare, "regression_count", 0)
    monkeypatch.setattr(nvbench_compare, "unknown_count", 0)


def compare_benches(nvbench_compare, ref_benches, cmp_benches, **kwargs):
    nvbench_compare.compare_benches(
        ref_benches,
        cmp_benches,
        threshold=kwargs.get("threshold", 0.0),
        plot_along=kwargs.get("plot_along"),
        plot=kwargs.get("plot", False),
        dark=False,
        axis_filters=kwargs.get("axis_filters", []),
        benchmark_filters=kwargs.get("benchmark_filters", []),
        no_color=True,
    )


def test_compare_benches_skips_non_finite_centers(monkeypatch, nvbench_compare):
    set_test_devices(monkeypatch, nvbench_compare)

    ref_benches = [
        make_benchmark(
            [
                make_state(nvbench_compare, "finite", mean="1.0"),
                make_state(nvbench_compare, "nan", mean="nan"),
                make_state(nvbench_compare, "inf", mean="inf"),
            ]
        )
    ]
    cmp_benches = [
        make_benchmark(
            [
                make_state(nvbench_compare, "finite", mean="1.0"),
                make_state(nvbench_compare, "nan", mean="1.0"),
                make_state(nvbench_compare, "inf", mean="1.0"),
            ]
        )
    ]

    compare_benches(nvbench_compare, ref_benches, cmp_benches)

    assert nvbench_compare.config_count == 1
    assert nvbench_compare.pass_count == 1
    assert nvbench_compare.improvement_count == 0
    assert nvbench_compare.regression_count == 0
    assert nvbench_compare.unknown_count == 0


def test_compare_benches_prefers_median_and_iqr_when_available(
    monkeypatch, nvbench_compare
):
    set_test_devices(monkeypatch, nvbench_compare)

    ref_state = make_state(nvbench_compare, "state", mean="1.0", noise="0.01")
    ref_state["summaries"].extend(
        [
            make_summary(nvbench_compare, "GPU_TIME_MEDIAN_TAG", "1.0"),
            make_summary(nvbench_compare, "GPU_TIME_IR_RELATIVE_TAG", "0.01"),
        ]
    )
    cmp_state = make_state(nvbench_compare, "state", mean="1.0", noise="0.01")
    cmp_state["summaries"].extend(
        [
            make_summary(nvbench_compare, "GPU_TIME_MEDIAN_TAG", "1.2"),
            make_summary(nvbench_compare, "GPU_TIME_IR_RELATIVE_TAG", "0.01"),
        ]
    )

    compare_benches(
        nvbench_compare, [make_benchmark([ref_state])], [make_benchmark([cmp_state])]
    )

    assert nvbench_compare.config_count == 1
    assert nvbench_compare.pass_count == 0
    assert nvbench_compare.improvement_count == 0
    assert nvbench_compare.regression_count == 1
    assert nvbench_compare.unknown_count == 0


def test_compare_benches_marks_unavailable_noise_unknown(monkeypatch, nvbench_compare):
    set_test_devices(monkeypatch, nvbench_compare)

    missing_noise_ref = make_state(nvbench_compare, "missing_noise")
    missing_noise_ref["summaries"] = [
        make_summary(nvbench_compare, "GPU_TIME_MEAN_TAG", "1.0")
    ]
    missing_noise_cmp = make_state(nvbench_compare, "missing_noise")
    missing_noise_cmp["summaries"] = [
        make_summary(nvbench_compare, "GPU_TIME_MEAN_TAG", "1.001")
    ]

    null_noise_ref = make_state(nvbench_compare, "null_noise")
    null_noise_ref["summaries"] = [
        make_summary(nvbench_compare, "GPU_TIME_MEAN_TAG", "1.0"),
        make_summary(nvbench_compare, "GPU_TIME_STDEV_RELATIVE_TAG", None),
    ]
    null_noise_cmp = make_state(nvbench_compare, "null_noise")
    null_noise_cmp["summaries"] = [
        make_summary(nvbench_compare, "GPU_TIME_MEAN_TAG", "1.001"),
        make_summary(nvbench_compare, "GPU_TIME_STDEV_RELATIVE_TAG", None),
    ]

    compare_benches(
        nvbench_compare,
        [make_benchmark([missing_noise_ref, null_noise_ref])],
        [make_benchmark([missing_noise_cmp, null_noise_cmp])],
    )

    assert nvbench_compare.config_count == 2
    assert nvbench_compare.pass_count == 0
    assert nvbench_compare.improvement_count == 0
    assert nvbench_compare.regression_count == 0
    assert nvbench_compare.unknown_count == 2


def test_plot_along_skips_states_without_selected_axis(monkeypatch, nvbench_compare):
    set_test_devices(monkeypatch, nvbench_compare)

    ref_benches = [
        make_benchmark(
            [
                make_state(nvbench_compare, "with_axis", axis_value=1),
                make_state(nvbench_compare, "without_axis"),
            ]
        )
    ]
    cmp_benches = [
        make_benchmark(
            [
                make_state(nvbench_compare, "with_axis", axis_value=1),
                make_state(nvbench_compare, "without_axis"),
            ]
        )
    ]

    compare_benches(
        nvbench_compare,
        ref_benches,
        cmp_benches,
        plot_along="A",
    )

    assert nvbench_compare.config_count == 2
    assert nvbench_compare.pass_count == 2
    assert nvbench_compare.improvement_count == 0
    assert nvbench_compare.regression_count == 0
    assert nvbench_compare.unknown_count == 0


def test_main_returns_success_exit_code_when_regressions_are_detected(
    monkeypatch, capsys, nvbench_compare
):
    devices = [{"id": 0, "name": "Test GPU"}]
    ref_root = {
        "devices": devices,
        "benchmarks": [
            make_benchmark([make_state(nvbench_compare, "state", mean="1.0")])
        ],
    }
    cmp_root = {
        "devices": devices,
        "benchmarks": [
            make_benchmark([make_state(nvbench_compare, "state", mean="1.2")])
        ],
    }

    def read_file(path):
        return ref_root if path == "ref.json" else cmp_root

    monkeypatch.setattr(nvbench_compare.reader, "read_file", read_file)
    monkeypatch.setattr(sys, "argv", ["nvbench_compare", "ref.json", "cmp.json"])

    assert nvbench_compare.main() == 0
    assert nvbench_compare.regression_count == 1
    assert (
        "Regression  (abs(%Diff) > max_noise, %Diff > 0): 1" in capsys.readouterr().out
    )
