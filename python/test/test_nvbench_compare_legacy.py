# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib.util
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def nvbench_compare_legacy(monkeypatch):
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
    monkeypatch.syspath_prepend(str(module_path.parent))
    spec = importlib.util.spec_from_file_location(
        "nvbench_compare_legacy_under_test", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    module.load_nvbench_compare_tooling()
    return module


def make_timing(nvbench_compare_legacy, mean, stdev_relative):
    return nvbench_compare_legacy.GpuTimingData(
        mean=mean,
        stdev=None,
        stdev_relative=stdev_relative,
    )


def make_state(nvbench_compare_legacy, name, *, mean="1.0", noise="0.01", axis_value=1):
    return {
        "name": name,
        "device": 0,
        "axis_values": [{"name": "A", "type": "int64", "value": axis_value}],
        "summaries": [
            {
                "tag": nvbench_compare_legacy.GPU_TIME_MEAN_TAG,
                "data": [{"name": "value", "type": "float64", "value": mean}],
            },
            {
                "tag": nvbench_compare_legacy.GPU_TIME_STDEV_RELATIVE_TAG,
                "data": [{"name": "value", "type": "float64", "value": noise}],
            },
        ],
    }


def make_benchmark(states):
    return {
        "name": "bench",
        "axes": [{"name": "A", "type": "int64", "flags": {}}],
        "devices": [0],
        "states": states,
    }


def make_run_data(nvbench_compare_legacy):
    devices = ({"id": 0, "name": "GPU"},)
    return nvbench_compare_legacy.ComparisonRunData(
        stats=nvbench_compare_legacy.ComparisonStats(),
        ref_devices=devices,
        cmp_devices=devices,
    )


def test_legacy_script_path_exposes_legacy_statuses_and_table(
    nvbench_compare_legacy,
):
    assert [status.value for status in nvbench_compare_legacy.ComparisonStatus] == [
        "????",
        "SAME",
        "FAST",
        "SLOW",
    ]
    assert nvbench_compare_legacy.get_display_headers()[0] == [
        "Ref Time",
        "Ref Noise",
        "Cmp Time",
        "Cmp Noise",
        "Diff",
        "%Diff",
        "Status",
    ]
    assert not hasattr(nvbench_compare_legacy, "ComparisonThresholds")


def test_legacy_compare_uses_mean_and_min_relative_stdev(nvbench_compare_legacy):
    same = nvbench_compare_legacy.compare_gpu_timings(
        make_timing(nvbench_compare_legacy, 1.0, 0.02),
        make_timing(nvbench_compare_legacy, 1.01, 0.03),
    )
    assert same.status == nvbench_compare_legacy.ComparisonStatus.SAME
    assert same.reason.code == "same"
    assert same.min_noise == pytest.approx(0.02)

    fast = nvbench_compare_legacy.compare_gpu_timings(
        make_timing(nvbench_compare_legacy, 1.0, 0.01),
        make_timing(nvbench_compare_legacy, 0.98, 0.01),
    )
    assert fast.status == nvbench_compare_legacy.ComparisonStatus.FAST
    assert fast.reason.code == "fast"

    slow = nvbench_compare_legacy.compare_gpu_timings(
        make_timing(nvbench_compare_legacy, 1.0, 0.01),
        make_timing(nvbench_compare_legacy, 1.02, 0.01),
    )
    assert slow.status == nvbench_compare_legacy.ComparisonStatus.SLOW
    assert slow.reason.code == "slow"


def test_legacy_compare_reports_unusable_noise_as_unknown(nvbench_compare_legacy):
    comparison = nvbench_compare_legacy.compare_gpu_timings(
        make_timing(nvbench_compare_legacy, 1.0, -0.01),
        make_timing(nvbench_compare_legacy, 1.01, 0.01),
    )

    assert comparison.status == nvbench_compare_legacy.ComparisonStatus.UNKNOWN
    assert comparison.reason.code == "noise_unavailable"


def test_legacy_plot_along_ignores_threshold_diff_table_filter(
    monkeypatch, nvbench_compare_legacy
):
    plot_calls = []

    def fake_plot(x, y, *args, **kwargs):
        plot_calls.append((list(x), list(y)))
        return [types.SimpleNamespace(get_color=lambda: "black")]

    monkeypatch.setattr(sys.modules["matplotlib.pyplot"], "plot", fake_plot)

    nvbench_compare_legacy.compare_benches(
        make_run_data(nvbench_compare_legacy),
        [
            make_benchmark(
                [
                    make_state(
                        nvbench_compare_legacy, "state", mean="1.0", axis_value=1
                    ),
                    make_state(
                        nvbench_compare_legacy, "state", mean="1.0", axis_value=2
                    ),
                ]
            )
        ],
        [
            make_benchmark(
                [
                    make_state(
                        nvbench_compare_legacy, "state", mean="1.01", axis_value=1
                    ),
                    make_state(
                        nvbench_compare_legacy, "state", mean="1.01", axis_value=2
                    ),
                ]
            )
        ],
        threshold=0.50,
        plot_along="A",
        plot=False,
        dark=False,
        filter_plan=nvbench_compare_legacy.build_benchmark_filter_plan([]),
        no_color=True,
    )

    assert plot_calls == [
        ([1.0, 2.0], [1.01, 1.01]),
        ([1.0, 2.0], [1.0, 1.0]),
    ]


def test_legacy_summary_plot_ignores_threshold_diff_table_filter(
    monkeypatch, nvbench_compare_legacy
):
    captured_entries = []

    def fake_plot_comparison_entries(entries, *args, **kwargs):
        captured_entries.extend(entries)

    monkeypatch.setattr(
        nvbench_compare_legacy,
        "plot_comparison_entries",
        fake_plot_comparison_entries,
    )

    nvbench_compare_legacy.compare_benches(
        make_run_data(nvbench_compare_legacy),
        [make_benchmark([make_state(nvbench_compare_legacy, "state", mean="1.0")])],
        [make_benchmark([make_state(nvbench_compare_legacy, "state", mean="1.01")])],
        threshold=0.50,
        plot_along=None,
        plot=True,
        dark=False,
        filter_plan=nvbench_compare_legacy.build_benchmark_filter_plan([]),
        no_color=True,
    )

    assert len(captured_entries) == 1
    assert captured_entries[0][1] == pytest.approx(0.01)


@pytest.mark.parametrize("threshold", ["nan", "inf", "-1"])
def test_legacy_main_rejects_invalid_threshold_diff(
    monkeypatch, capsys, nvbench_compare_legacy, threshold
):
    monkeypatch.setattr(
        nvbench_compare_legacy,
        "load_nvbench_compare_tooling",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("tooling should not load")
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nvbench_compare",
            "--threshold-diff",
            threshold,
            "ref.json",
            "cmp.json",
        ],
    )

    assert nvbench_compare_legacy.main() == 1
    assert (
        "--threshold-diff must be a finite non-negative percentage"
        in capsys.readouterr().out
    )
