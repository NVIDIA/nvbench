# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
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


def make_binary_summary(nvbench_compare, tag, filename, size):
    return {
        "tag": getattr(nvbench_compare, tag),
        "data": [
            {"name": "filename", "type": "string", "value": filename},
            {"name": "size", "type": "int64", "value": str(size)},
        ],
    }


def make_gpu_timing_data(
    nvbench_compare,
    *,
    mean=1.0,
    stdev=None,
    stdev_relative=0.01,
    median=None,
    interquartile_range=None,
    interquartile_range_relative=None,
):
    return nvbench_compare.GpuTimingData(
        minimum=None,
        maximum=None,
        mean=mean,
        stdev=stdev,
        stdev_relative=stdev_relative,
        median=median,
        interquartile_range=interquartile_range,
        interquartile_range_relative=interquartile_range_relative,
    )


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


def make_comparison_run_data(nvbench_compare, ref_devices=None, cmp_devices=None):
    devices = [{"id": 0, "name": "Test GPU"}]
    return nvbench_compare.ComparisonRunData(
        stats=nvbench_compare.ComparisonStats(),
        ref_devices=tuple(devices if ref_devices is None else ref_devices),
        cmp_devices=tuple(devices if cmp_devices is None else cmp_devices),
    )


def make_filter_plan(nvbench_compare, filter_actions=None):
    return nvbench_compare.build_benchmark_filter_plan(filter_actions or [])


def test_compare_benches_accepts_matching_duplicate_state_counts(
    monkeypatch, nvbench_compare
):
    run_data = make_comparison_run_data(nvbench_compare)

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
        run_data,
        ref_benches,
        cmp_benches,
        threshold=0.0,
        plot_along=None,
        plot=False,
        dark=False,
        filter_plan=make_filter_plan(nvbench_compare),
        no_color=True,
    )

    assert run_data.stats.config_count == 3
    assert run_data.stats.pass_count == 3
    assert run_data.stats.improvement_count == 0
    assert run_data.stats.regression_count == 0
    assert run_data.stats.unknown_count == 0


def test_compare_benches_rejects_swapped_duplicate_state_counts(
    monkeypatch, nvbench_compare
):
    run_data = make_comparison_run_data(nvbench_compare)

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
            run_data,
            ref_benches,
            cmp_benches,
            threshold=0.0,
            plot_along=None,
            plot=False,
            dark=False,
            filter_plan=make_filter_plan(nvbench_compare),
            no_color=True,
        )


def test_compare_benches_matches_duplicate_states_after_axis_filter(
    monkeypatch, nvbench_compare
):
    run_data = make_comparison_run_data(nvbench_compare)

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
        run_data,
        ref_benches,
        cmp_benches,
        threshold=0.0,
        plot_along=None,
        plot=False,
        dark=False,
        filter_plan=make_filter_plan(nvbench_compare, [("axis", "A=2")]),
        no_color=True,
    )

    assert run_data.stats.config_count == 1
    assert run_data.stats.pass_count == 1
    assert run_data.stats.improvement_count == 0
    assert run_data.stats.regression_count == 0
    assert run_data.stats.unknown_count == 0


def test_compare_benches_skips_non_finite_centers(monkeypatch, nvbench_compare):
    run_data = make_comparison_run_data(nvbench_compare)

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

    nvbench_compare.compare_benches(
        run_data,
        ref_benches,
        cmp_benches,
        threshold=0.0,
        plot_along=None,
        plot=False,
        dark=False,
        filter_plan=make_filter_plan(nvbench_compare),
        no_color=True,
    )

    assert run_data.stats.config_count == 1
    assert run_data.stats.pass_count == 1
    assert run_data.stats.improvement_count == 0
    assert run_data.stats.regression_count == 0
    assert run_data.stats.unknown_count == 0


def test_gpu_timing_data_loads_samples_and_frequencies(tmp_path, nvbench_compare):
    samples_dir = tmp_path / "result.json-bin"
    freqs_dir = tmp_path / "result.json-freqs-bin"
    samples_dir.mkdir()
    freqs_dir.mkdir()
    samples_file = samples_dir / "0.bin"
    freqs_file = freqs_dir / "0.bin"

    np.array([1.0, 2.0, 4.0], dtype="<f4").tofile(samples_file)
    np.array([100.0, 200.0, 400.0], dtype="<f4").tofile(freqs_file)

    timing = nvbench_compare.extract_gpu_timing_data(
        [
            make_summary(nvbench_compare, "GPU_TIME_MEAN_TAG", "2.0"),
            make_binary_summary(
                nvbench_compare,
                "SAMPLE_TIMES_TAG",
                str(samples_file.relative_to(tmp_path)),
                3,
            ),
            make_binary_summary(
                nvbench_compare,
                "SAMPLE_FREQUENCIES_TAG",
                str(freqs_file.relative_to(tmp_path)),
                3,
            ),
        ],
        str(tmp_path),
    )

    assert timing.samples is not None
    assert list(timing.samples) == pytest.approx([1.0, 2.0, 4.0])
    assert timing.frequencies is not None
    assert list(timing.frequencies) == pytest.approx([100.0, 200.0, 400.0])


def test_gpu_timing_data_rejects_mismatched_sample_and_frequency_counts(
    tmp_path, nvbench_compare
):
    samples_file = tmp_path / "samples.bin"
    freqs_file = tmp_path / "freqs.bin"
    np.array([1.0, 2.0], dtype="<f4").tofile(samples_file)
    np.array([100.0, 200.0, 300.0], dtype="<f4").tofile(freqs_file)

    with pytest.raises(ValueError, match="sample count .* frequency count"):
        nvbench_compare.extract_gpu_timing_data(
            [
                make_binary_summary(
                    nvbench_compare, "SAMPLE_TIMES_TAG", str(samples_file), 2
                ),
                make_binary_summary(
                    nvbench_compare, "SAMPLE_FREQUENCIES_TAG", str(freqs_file), 3
                ),
            ],
            str(tmp_path),
        )


def test_compare_gpu_timings_classifies_common_cases(nvbench_compare):
    ref_timing = make_gpu_timing_data(nvbench_compare, mean=1.0, stdev_relative=0.05)

    same = nvbench_compare.compare_gpu_timings(
        ref_timing,
        make_gpu_timing_data(nvbench_compare, mean=1.03, stdev_relative=0.05),
    )
    assert same is not None
    assert same.status == nvbench_compare.ComparisonStatus.SAME
    assert same.ref_time == pytest.approx(1.0)
    assert same.cmp_time == pytest.approx(1.03)
    assert same.diff == pytest.approx(0.03)
    assert same.frac_diff == pytest.approx(0.03)
    assert same.max_noise == pytest.approx(0.05)

    fast = nvbench_compare.compare_gpu_timings(
        ref_timing,
        make_gpu_timing_data(nvbench_compare, mean=0.8, stdev_relative=0.05),
    )
    assert fast is not None
    assert fast.status == nvbench_compare.ComparisonStatus.FAST

    slow = nvbench_compare.compare_gpu_timings(
        ref_timing,
        make_gpu_timing_data(nvbench_compare, mean=1.2, stdev_relative=0.05),
    )
    assert slow is not None
    assert slow.status == nvbench_compare.ComparisonStatus.SLOW

    unknown = nvbench_compare.compare_gpu_timings(
        ref_timing,
        make_gpu_timing_data(nvbench_compare, mean=1.2, stdev_relative=None),
    )
    assert unknown is not None
    assert unknown.status == nvbench_compare.ComparisonStatus.UNKNOWN


@pytest.mark.parametrize("ref_time, cmp_time", [(None, 1.0), (1.0, None), (0.0, 1.0)])
def test_compare_gpu_timings_rejects_unusable_centers(
    nvbench_compare, ref_time, cmp_time
):
    assert (
        nvbench_compare.compare_gpu_timings(
            make_gpu_timing_data(nvbench_compare, mean=ref_time),
            make_gpu_timing_data(nvbench_compare, mean=cmp_time),
        )
        is None
    )


def test_compare_benches_prefers_median_and_iqr_when_available(
    monkeypatch, nvbench_compare
):
    run_data = make_comparison_run_data(nvbench_compare)

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

    nvbench_compare.compare_benches(
        run_data,
        [make_benchmark([ref_state])],
        [make_benchmark([cmp_state])],
        threshold=0.0,
        plot_along=None,
        plot=False,
        dark=False,
        filter_plan=make_filter_plan(nvbench_compare),
        no_color=True,
    )

    assert run_data.stats.config_count == 1
    assert run_data.stats.pass_count == 0
    assert run_data.stats.improvement_count == 0
    assert run_data.stats.regression_count == 1
    assert run_data.stats.unknown_count == 0


def test_compare_benches_marks_unavailable_noise_unknown(monkeypatch, nvbench_compare):
    run_data = make_comparison_run_data(nvbench_compare)

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

    nvbench_compare.compare_benches(
        run_data,
        [make_benchmark([missing_noise_ref, null_noise_ref])],
        [make_benchmark([missing_noise_cmp, null_noise_cmp])],
        threshold=0.0,
        plot_along=None,
        plot=False,
        dark=False,
        filter_plan=make_filter_plan(nvbench_compare),
        no_color=True,
    )

    assert run_data.stats.config_count == 2
    assert run_data.stats.pass_count == 0
    assert run_data.stats.improvement_count == 0
    assert run_data.stats.regression_count == 0
    assert run_data.stats.unknown_count == 2


def test_plot_along_skips_states_without_selected_axis(monkeypatch, nvbench_compare):
    run_data = make_comparison_run_data(nvbench_compare)

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

    nvbench_compare.compare_benches(
        run_data,
        ref_benches,
        cmp_benches,
        threshold=0.0,
        plot_along="A",
        plot=False,
        dark=False,
        filter_plan=make_filter_plan(nvbench_compare),
        no_color=True,
    )

    assert run_data.stats.config_count == 2
    assert run_data.stats.pass_count == 2
    assert run_data.stats.improvement_count == 0
    assert run_data.stats.regression_count == 0
    assert run_data.stats.unknown_count == 0


def test_device_filter_parser_accepts_all_and_duplicate_ids(nvbench_compare):
    assert nvbench_compare.parse_device_filter(" all ", "--reference-devices") is None
    assert nvbench_compare.parse_device_filter("0", "--reference-devices") == [0]
    assert nvbench_compare.parse_device_filter("0, 2,0", "--reference-devices") == [
        0,
        2,
        0,
    ]


@pytest.mark.parametrize(
    "device_arg",
    [
        "",
        " ",
        "gpu",
        "-1",
        "0,gpu",
        "0,-1",
        "0,",
        ",0",
    ],
)
def test_device_filter_parser_rejects_invalid_values(nvbench_compare, device_arg):
    with pytest.raises(ValueError, match="must be 'all'"):
        nvbench_compare.parse_device_filter(device_arg, "--reference-devices")


def test_explicit_device_filters_downgrade_device_mismatch_to_warning(nvbench_compare):
    assert nvbench_compare.require_matching_device_sections(None, None)
    assert not nvbench_compare.require_matching_device_sections([0], None)
    assert not nvbench_compare.require_matching_device_sections(None, [1])
    assert not nvbench_compare.require_matching_device_sections([0], [1])


def test_compare_benches_pairs_filtered_devices_by_position(
    monkeypatch, nvbench_compare
):
    run_data = make_comparison_run_data(
        nvbench_compare,
        ref_devices=[
            {"id": 0, "name": "Reference GPU 0"},
            {"id": 1, "name": "Reference GPU 1"},
        ],
        cmp_devices=[
            {"id": 0, "name": "Compare GPU 0"},
            {"id": 1, "name": "Compare GPU 1"},
        ],
    )

    ref_benches = [
        make_benchmark(
            [
                make_state(nvbench_compare, "Device=0", mean="1.0", device=0),
                make_state(nvbench_compare, "Device=1", mean="9.0", device=1),
            ]
        )
    ]
    cmp_benches = [
        make_benchmark(
            [
                make_state(nvbench_compare, "Device=0", mean="9.0", device=0),
                make_state(nvbench_compare, "Device=1", mean="1.0", device=1),
            ]
        )
    ]

    nvbench_compare.compare_benches(
        run_data,
        ref_benches,
        cmp_benches,
        threshold=0.0,
        plot_along=None,
        plot=False,
        dark=False,
        filter_plan=make_filter_plan(nvbench_compare),
        no_color=True,
        reference_device_filter=[0],
        compare_device_filter=[1],
    )

    assert run_data.stats.config_count == 1
    assert run_data.stats.pass_count == 1
    assert run_data.stats.improvement_count == 0
    assert run_data.stats.regression_count == 0
    assert run_data.stats.unknown_count == 0


def test_axis_filter_applies_to_most_recent_benchmark(monkeypatch, nvbench_compare):
    run_data = make_comparison_run_data(nvbench_compare)

    ref_benches = [
        make_benchmark(
            [
                make_state(nvbench_compare, "state", mean="1.0", axis_value=1),
                make_state(nvbench_compare, "state", mean="2.0", axis_value=2),
            ],
            name="bench1",
        ),
        make_benchmark(
            [
                make_state(nvbench_compare, "state", mean="3.0", axis_value=1),
                make_state(nvbench_compare, "state", mean="4.0", axis_value=2),
            ],
            name="bench2",
        ),
    ]
    cmp_benches = [
        make_benchmark(
            [
                make_state(nvbench_compare, "state", mean="1.0", axis_value=1),
                make_state(nvbench_compare, "state", mean="2.0", axis_value=2),
            ],
            name="bench1",
        ),
        make_benchmark(
            [
                make_state(nvbench_compare, "state", mean="3.0", axis_value=1),
                make_state(nvbench_compare, "state", mean="4.0", axis_value=2),
            ],
            name="bench2",
        ),
    ]

    nvbench_compare.compare_benches(
        run_data,
        ref_benches,
        cmp_benches,
        threshold=0.0,
        plot_along=None,
        plot=False,
        dark=False,
        filter_plan=make_filter_plan(
            nvbench_compare,
            [("benchmark", "bench1"), ("axis", "A=2"), ("benchmark", "bench2")],
        ),
        no_color=True,
    )

    assert run_data.stats.config_count == 3
    assert run_data.stats.pass_count == 3
    assert run_data.stats.improvement_count == 0
    assert run_data.stats.regression_count == 0
    assert run_data.stats.unknown_count == 0


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
    assert (
        "Regression  (abs(%Diff) > max_noise, %Diff > 0): 1" in capsys.readouterr().out
    )
