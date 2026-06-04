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
    minimum=None,
    maximum=None,
    mean=1.0,
    stdev=None,
    stdev_relative=0.01,
    first_quartile=None,
    median=None,
    third_quartile=None,
    interquartile_range=None,
    interquartile_range_relative=None,
    sm_clock_rate_mean=None,
    sample_values=None,
    frequency_values=None,
):
    return nvbench_compare.GpuTimingData(
        minimum=minimum,
        maximum=maximum,
        mean=mean,
        stdev=stdev,
        stdev_relative=stdev_relative,
        first_quartile=first_quartile,
        median=median,
        third_quartile=third_quartile,
        interquartile_range=interquartile_range,
        interquartile_range_relative=interquartile_range_relative,
        sm_clock_rate_mean=sm_clock_rate_mean,
        sample_source=None
        if sample_values is None
        else types.SimpleNamespace(values=np.asarray(sample_values, dtype=np.float32)),
        frequency_source=None
        if frequency_values is None
        else types.SimpleNamespace(
            values=np.asarray(frequency_values, dtype=np.float32)
        ),
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
    assert run_data.stats.pass_count == 0
    assert run_data.stats.improvement_count == 0
    assert run_data.stats.regression_count == 0
    assert run_data.stats.undecided_count == 3
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
    assert run_data.stats.pass_count == 0
    assert run_data.stats.improvement_count == 0
    assert run_data.stats.regression_count == 0
    assert run_data.stats.undecided_count == 1
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
    assert run_data.stats.pass_count == 0
    assert run_data.stats.improvement_count == 0
    assert run_data.stats.regression_count == 0
    assert run_data.stats.undecided_count == 1
    assert run_data.stats.unknown_count == 0


def test_gpu_timing_data_loads_samples_and_frequencies_lazily(
    tmp_path, nvbench_compare
):
    samples_dir = tmp_path / "result.json-bin"
    freqs_dir = tmp_path / "result.json-freqs-bin"
    samples_dir.mkdir()
    freqs_dir.mkdir()
    samples_file = samples_dir / "0.bin"
    freqs_file = freqs_dir / "0.bin"

    np.array([1.0, 2.0, 4.0], dtype="<f4").tofile(samples_file)
    np.array([100.0, 200.0, 400.0], dtype="<f4").tofile(freqs_file)

    reader_calls = []
    buffers = {
        str(samples_file): np.array([1.0, 2.0, 4.0], dtype="<f4").tobytes(),
        str(freqs_file): np.array([100.0, 200.0, 400.0], dtype="<f4").tobytes(),
    }

    def tracking_reader(filename):
        reader_calls.append(filename)
        return buffers[filename]

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
        float32_reader=tracking_reader,
    )

    assert reader_calls == []

    assert timing.samples is not None
    assert list(timing.samples) == pytest.approx([1.0, 2.0, 4.0])
    assert reader_calls == [str(samples_file)]

    assert list(timing.samples) == pytest.approx([1.0, 2.0, 4.0])
    assert reader_calls == [str(samples_file)]

    assert timing.frequencies is not None
    assert list(timing.frequencies) == pytest.approx([100.0, 200.0, 400.0])
    assert reader_calls == [str(samples_file), str(freqs_file)]


def test_gpu_timing_data_parses_quartiles_and_sm_clock_rate_mean(nvbench_compare):
    timing = nvbench_compare.extract_gpu_timing_data(
        [
            make_summary(nvbench_compare, "GPU_TIME_MEAN_TAG", "2.0"),
            make_summary(nvbench_compare, "GPU_TIME_Q1_TAG", "1.5"),
            make_summary(nvbench_compare, "GPU_TIME_MEDIAN_TAG", "2.0"),
            make_summary(nvbench_compare, "GPU_TIME_Q3_TAG", "2.5"),
            make_summary(nvbench_compare, "GPU_SM_CLOCK_RATE_MEAN_TAG", "1.5e9"),
        ],
    )

    assert timing.first_quartile == pytest.approx(1.5)
    assert timing.median == pytest.approx(2.0)
    assert timing.third_quartile == pytest.approx(2.5)
    assert timing.sm_clock_rate_mean == pytest.approx(1.5e9)
    assert timing.frequencies is None


def test_gpu_timing_data_treats_mismatched_sample_and_frequency_counts_as_unavailable(
    tmp_path, nvbench_compare
):
    samples_file = tmp_path / "samples.bin"
    freqs_file = tmp_path / "freqs.bin"
    np.array([1.0, 2.0], dtype="<f4").tofile(samples_file)
    np.array([100.0, 200.0, 300.0], dtype="<f4").tofile(freqs_file)

    with pytest.warns(RuntimeWarning, match="sample count .* frequency count"):
        timing = nvbench_compare.extract_gpu_timing_data(
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

    assert timing.samples is None
    assert timing.frequencies is None


def test_gpu_timing_data_warns_when_lazy_sample_read_fails(tmp_path, nvbench_compare):
    missing_file = tmp_path / "missing.bin"

    timing = nvbench_compare.extract_gpu_timing_data(
        [
            make_binary_summary(
                nvbench_compare, "SAMPLE_TIMES_TAG", str(missing_file), 3
            ),
        ],
        str(tmp_path),
    )

    with pytest.warns(RuntimeWarning, match="failed to read"):
        assert timing.samples is None

    assert timing.samples is None


def test_compare_gpu_timings_classifies_common_cases(nvbench_compare):
    ref_timing = make_gpu_timing_data(nvbench_compare, mean=1.0, stdev_relative=0.05)

    undecided = nvbench_compare.compare_gpu_timings(
        ref_timing,
        make_gpu_timing_data(nvbench_compare, mean=1.03, stdev_relative=0.05),
    )
    assert undecided is not None
    assert undecided.status == nvbench_compare.ComparisonStatus.UNDECIDED
    assert undecided.ref_time == pytest.approx(1.0)
    assert undecided.cmp_time == pytest.approx(1.03)
    assert undecided.diff == pytest.approx(0.03)
    assert undecided.frac_diff == pytest.approx(0.03)
    assert undecided.max_noise == pytest.approx(0.05)
    assert undecided.reason.code == "noise_too_high"

    ref_interval_timing = make_gpu_timing_data(
        nvbench_compare,
        minimum=1.0,
        first_quartile=1.1,
        median=1.2,
        third_quartile=1.3,
        mean=1.2,
        stdev_relative=0.05,
        interquartile_range_relative=0.01,
        sm_clock_rate_mean=100.0,
    )

    fast = nvbench_compare.compare_gpu_timings(
        ref_interval_timing,
        make_gpu_timing_data(
            nvbench_compare,
            minimum=0.8,
            first_quartile=0.85,
            median=0.9,
            third_quartile=0.95,
            mean=0.9,
            stdev_relative=0.05,
            sm_clock_rate_mean=100.0,
        ),
    )
    assert fast is not None
    assert fast.status == nvbench_compare.ComparisonStatus.FAST
    assert fast.reason.code == "clear_gap_confirmed_by_summary_cycles"
    assert fast.diff_interval == pytest.approx((-0.5, -0.05))
    assert fast.frac_diff_interval == pytest.approx((-0.3846153846, -0.05))

    slow = nvbench_compare.compare_gpu_timings(
        ref_interval_timing,
        make_gpu_timing_data(
            nvbench_compare,
            minimum=1.4,
            first_quartile=1.45,
            median=1.5,
            third_quartile=1.55,
            mean=1.5,
            stdev_relative=0.05,
            sm_clock_rate_mean=100.0,
        ),
    )
    assert slow is not None
    assert slow.status == nvbench_compare.ComparisonStatus.SLOW
    assert slow.reason.code == "clear_gap_confirmed_by_summary_cycles"
    assert slow.diff_interval == pytest.approx((0.1, 0.55))
    assert slow.frac_diff_interval == pytest.approx((0.0769230769, 0.55))

    same = nvbench_compare.compare_gpu_timings(
        ref_interval_timing,
        make_gpu_timing_data(
            nvbench_compare,
            minimum=1.02,
            first_quartile=1.1,
            median=1.204,
            third_quartile=1.28,
            mean=1.204,
            interquartile_range_relative=0.01,
            sm_clock_rate_mean=100.0,
        ),
    )
    assert same is not None
    assert same.status == nvbench_compare.ComparisonStatus.SAME
    assert same.reason.code == "same_confirmed_by_cycles"
    assert same.diff_interval == pytest.approx((-0.28, 0.28))
    assert same.frac_diff_interval == pytest.approx((-0.2153846154, 0.28))

    weak_overlap = nvbench_compare.compare_gpu_timings(
        make_gpu_timing_data(
            nvbench_compare,
            minimum=1.0,
            first_quartile=1.19,
            median=1.195,
            third_quartile=1.2,
            mean=1.195,
            interquartile_range_relative=0.01,
        ),
        make_gpu_timing_data(
            nvbench_compare,
            minimum=1.2,
            first_quartile=1.2,
            median=1.2,
            third_quartile=1.4,
            mean=1.2,
            interquartile_range_relative=0.01,
        ),
    )
    assert weak_overlap is not None
    assert weak_overlap.status == nvbench_compare.ComparisonStatus.UNDECIDED
    assert weak_overlap.reason.code == "weak_interval_overlap"

    center_too_far = nvbench_compare.compare_gpu_timings(
        ref_interval_timing,
        make_gpu_timing_data(
            nvbench_compare,
            minimum=1.0,
            first_quartile=1.1,
            median=1.21,
            third_quartile=1.3,
            mean=1.21,
            interquartile_range_relative=0.01,
        ),
    )
    assert center_too_far is not None
    assert center_too_far.status == nvbench_compare.ComparisonStatus.UNDECIDED
    assert center_too_far.reason.code == "centers_not_close"

    noisy_same = nvbench_compare.compare_gpu_timings(
        ref_interval_timing,
        make_gpu_timing_data(
            nvbench_compare,
            minimum=1.02,
            first_quartile=1.1,
            median=1.204,
            third_quartile=1.28,
            mean=1.204,
            interquartile_range_relative=0.03,
        ),
    )
    assert noisy_same is not None
    assert noisy_same.status == nvbench_compare.ComparisonStatus.UNDECIDED
    assert noisy_same.reason.code == "noise_too_high"

    clock_disagreement = nvbench_compare.compare_gpu_timings(
        ref_interval_timing,
        make_gpu_timing_data(
            nvbench_compare,
            minimum=1.02,
            first_quartile=1.1,
            median=1.204,
            third_quartile=1.28,
            mean=1.204,
            interquartile_range_relative=0.01,
            sm_clock_rate_mean=200.0,
        ),
    )
    assert clock_disagreement is not None
    assert clock_disagreement.status == nvbench_compare.ComparisonStatus.UNDECIDED
    assert clock_disagreement.reason.code == "cycle_same_not_confirmed"

    missing_clock = nvbench_compare.compare_gpu_timings(
        ref_interval_timing,
        make_gpu_timing_data(
            nvbench_compare,
            minimum=0.8,
            first_quartile=0.85,
            median=0.9,
            third_quartile=0.95,
            mean=0.9,
            stdev_relative=0.05,
        ),
    )
    assert missing_clock is not None
    assert missing_clock.status == nvbench_compare.ComparisonStatus.UNDECIDED
    assert missing_clock.reason.code == "missing_clock_rate"

    frequency_shift = nvbench_compare.compare_gpu_timings(
        ref_interval_timing,
        make_gpu_timing_data(
            nvbench_compare,
            minimum=0.8,
            first_quartile=0.85,
            median=0.9,
            third_quartile=0.95,
            mean=0.9,
            stdev_relative=0.05,
            sm_clock_rate_mean=200.0,
        ),
    )
    assert frequency_shift is not None
    assert frequency_shift.status == nvbench_compare.ComparisonStatus.UNDECIDED
    assert frequency_shift.reason.code == "summary_cycle_gap_not_confirmed"

    bulk_cycle_fast = nvbench_compare.compare_gpu_timings(
        make_gpu_timing_data(
            nvbench_compare,
            minimum=1.0,
            first_quartile=1.1,
            median=1.2,
            third_quartile=1.3,
            mean=1.2,
            stdev_relative=0.05,
            sample_values=[1.0, 1.1, 1.2, 1.3],
            frequency_values=[100.0] * 4,
        ),
        make_gpu_timing_data(
            nvbench_compare,
            minimum=0.8,
            first_quartile=0.85,
            median=0.9,
            third_quartile=0.95,
            mean=0.9,
            stdev_relative=0.05,
            sample_values=[0.8, 0.85, 0.9, 0.95],
            frequency_values=[100.0] * 4,
        ),
    )
    assert bulk_cycle_fast is not None
    assert bulk_cycle_fast.status == nvbench_compare.ComparisonStatus.FAST
    assert bulk_cycle_fast.reason.code == "clear_gap_confirmed_by_bulk_cycles"

    bulk_cycle_shift = nvbench_compare.compare_gpu_timings(
        make_gpu_timing_data(
            nvbench_compare,
            minimum=1.0,
            first_quartile=1.1,
            median=1.2,
            third_quartile=1.3,
            mean=1.2,
            stdev_relative=0.05,
            sample_values=[1.0, 1.1, 1.2, 1.3],
            frequency_values=[100.0] * 4,
        ),
        make_gpu_timing_data(
            nvbench_compare,
            minimum=0.8,
            first_quartile=0.85,
            median=0.9,
            third_quartile=0.95,
            mean=0.9,
            stdev_relative=0.05,
            sample_values=[0.8, 0.85, 0.9, 0.95],
            frequency_values=[200.0] * 4,
        ),
    )
    assert bulk_cycle_shift is not None
    assert bulk_cycle_shift.status == nvbench_compare.ComparisonStatus.UNDECIDED
    assert bulk_cycle_shift.reason.code == "bulk_cycle_gap_not_confirmed"

    missing_noise = nvbench_compare.compare_gpu_timings(
        ref_timing,
        make_gpu_timing_data(nvbench_compare, mean=1.2, stdev_relative=None),
    )
    assert missing_noise is not None
    assert missing_noise.status == nvbench_compare.ComparisonStatus.UNDECIDED
    assert missing_noise.max_noise is None
    assert missing_noise.reason.code == "noise_unavailable"


def test_compare_gpu_timings_uses_bulk_data_to_confirm_same(nvbench_compare):
    ref_timing = make_gpu_timing_data(
        nvbench_compare,
        mean=1.0,
        stdev_relative=0.05,
        sample_values=[1.0] * 8 + [1.004] * 2,
        frequency_values=[100.0] * 10,
    )
    cmp_timing = make_gpu_timing_data(
        nvbench_compare,
        mean=1.0,
        stdev_relative=0.05,
        sample_values=[1.0] * 2 + [1.004] * 8,
        frequency_values=[100.0] * 10,
    )

    comparison = nvbench_compare.compare_gpu_timings(ref_timing, cmp_timing)

    assert comparison is not None
    assert comparison.status == nvbench_compare.ComparisonStatus.SAME
    assert comparison.reason.code == "bulk_same"


def test_format_diff_and_percent_ranges(nvbench_compare):
    assert nvbench_compare.format_duration_range((-12e-6, 8e-6)) == "[-12.00, 8.00] us"
    assert (
        nvbench_compare.format_percentage_bounds(
            (-0.2153846154, 0.28), nvbench_compare.ComparisonStatus.UNDECIDED
        )
        == "in [-21.5%, +28.0%]"
    )
    assert (
        nvbench_compare.format_percentage_bounds(
            (-0.3076923077, -0.05), nvbench_compare.ComparisonStatus.FAST
        )
        == "<= -5.0%"
    )
    assert (
        nvbench_compare.format_percentage_bounds(
            (0.0769230769, 0.55), nvbench_compare.ComparisonStatus.SLOW
        )
        == ">= +7.7%"
    )


def test_format_timing_with_interval(nvbench_compare):
    interval = nvbench_compare.TimingInterval(
        lower=0.002237, upper=0.002389, center=0.0023
    )
    assert (
        nvbench_compare.format_timing_with_interval(0.0023, interval)
        == "2.300 ms [-63, +89] us"
    )


def test_format_timing_with_explicit_interval(nvbench_compare):
    interval = nvbench_compare.TimingInterval(
        lower=0.001434, upper=0.001458, center=0.001446
    )
    assert (
        nvbench_compare.format_timing_with_explicit_interval(0.001446, interval)
        == "1.4[34 | 46 | 58] ms"
    )


def test_compare_gpu_timings_keeps_bulk_mismatch_undecided(nvbench_compare):
    ref_timing = make_gpu_timing_data(
        nvbench_compare,
        minimum=1.0,
        first_quartile=1.1,
        median=1.2,
        third_quartile=1.3,
        mean=1.2,
        interquartile_range_relative=0.01,
        sample_values=[1.0, 1.0, 1.004, 1.004],
        frequency_values=[100.0] * 4,
    )
    cmp_timing = make_gpu_timing_data(
        nvbench_compare,
        minimum=1.02,
        first_quartile=1.1,
        median=1.204,
        third_quartile=1.28,
        mean=1.204,
        interquartile_range_relative=0.01,
        sample_values=[1.02, 1.02, 1.024, 1.024],
        frequency_values=[100.0] * 4,
    )

    comparison = nvbench_compare.compare_gpu_timings(ref_timing, cmp_timing)

    assert comparison is not None
    assert comparison.status == nvbench_compare.ComparisonStatus.UNDECIDED
    assert comparison.reason.code == "bulk_time_support_mismatch"
    assert "sample: min(ref=0.0%, cmp=0.0%) >= 99.0%" in comparison.reason.message
    assert "support: min(ref=0.0%, cmp=0.0%) >= 80.0%" in comparison.reason.message
    assert "99.0%" in comparison.reason.message
    assert "80.0%" in comparison.reason.message


def test_compare_gpu_timings_requires_bulk_cycle_coverage(nvbench_compare):
    ref_timing = make_gpu_timing_data(
        nvbench_compare,
        mean=1.0,
        stdev_relative=0.01,
        sample_values=[1.0, 1.0, 1.004, 1.004],
        frequency_values=[100.0] * 4,
    )
    cmp_timing = make_gpu_timing_data(
        nvbench_compare,
        mean=1.0,
        stdev_relative=0.01,
        sample_values=[1.0, 1.0, 1.004, 1.004],
        frequency_values=[200.0] * 4,
    )

    comparison = nvbench_compare.compare_gpu_timings(ref_timing, cmp_timing)

    assert comparison is not None
    assert comparison.status == nvbench_compare.ComparisonStatus.UNDECIDED
    assert comparison.reason.code == "bulk_cycle_support_mismatch"


def test_bulk_same_reports_sample_weight_coverage_mismatch(nvbench_compare):
    ref_values = [1.0, 1.001, 1.002, 1.003] + [1.02] * 100
    cmp_values = [1.0, 1.001, 1.002, 1.003]

    decision = nvbench_compare.compare_values_for_bulk_same(
        ref_values,
        cmp_values,
        label="time",
        thresholds=nvbench_compare.ComparisonThresholds(),
    )

    assert decision.status == nvbench_compare.ComparisonStatus.UNDECIDED
    assert decision.reason.code == "bulk_time_support_mismatch"
    assert "sample: min(ref=3.8%, cmp=100.0%) >= 99.0%" in decision.reason.message
    assert "support: min(ref=80.0%, cmp=100.0%) >= 80.0%" in decision.reason.message


def test_bulk_same_filters_rare_values_from_support_coverage(nvbench_compare):
    ref_values = [1.0] * 1000 + [1.02 + 0.01 * i for i in range(10)]
    cmp_values = [1.0]

    decision = nvbench_compare.compare_values_for_bulk_same(
        ref_values,
        cmp_values,
        label="time",
        thresholds=nvbench_compare.ComparisonThresholds(),
    )

    assert decision.status == nvbench_compare.ComparisonStatus.SAME
    assert decision.reason.code == "bulk_time_same"


def test_bulk_same_reports_unique_support_coverage_mismatch(nvbench_compare):
    ref_values = [1.0] * 1000 + [1.02 + 0.01 * i for i in range(10)]
    cmp_values = [1.0]

    decision = nvbench_compare.compare_values_for_bulk_same(
        ref_values,
        cmp_values,
        label="time",
        thresholds=nvbench_compare.ComparisonThresholds(
            bulk_support_max_removed_sample_fraction=0.005
        ),
    )

    assert decision.status == nvbench_compare.ComparisonStatus.UNDECIDED
    assert decision.reason.code == "bulk_time_support_mismatch"
    assert "sample: min(ref=99.0%, cmp=100.0%) >= 99.0%" in decision.reason.message
    assert "support: min(ref=9.1%, cmp=100.0%) >= 80.0%" in decision.reason.message


def test_bulk_same_retains_full_support_when_all_values_are_unique(nvbench_compare):
    coverages = nvbench_compare.compute_nearest_neighbor_coverages(
        [1.0, 1.02],
        [1.0],
        thresholds=nvbench_compare.ComparisonThresholds(
            bulk_support_rare_sample_fraction=1.0,
            bulk_support_max_removed_sample_fraction=1.0,
        ),
    )

    assert coverages is not None
    assert coverages["ref_sample"] == 0.5
    assert coverages["ref_support"] == 0.5
    assert coverages["ref_support_filter"] == nvbench_compare.SupportFilterInfo(
        activated=False,
        reason="all_values_unique",
        removed_sample_fraction=0.0,
    )


def test_comparison_stats_records_undecided_status(nvbench_compare):
    stats = nvbench_compare.ComparisonStats()

    stats.record(nvbench_compare.ComparisonStatus.UNDECIDED)

    assert stats.config_count == 1
    assert stats.pass_count == 0
    assert stats.improvement_count == 0
    assert stats.regression_count == 0
    assert stats.undecided_count == 1
    assert stats.unknown_count == 0


def test_comparison_stats_records_undecided_reason(nvbench_compare):
    stats = nvbench_compare.ComparisonStats()
    less_severe_reason = nvbench_compare.DecisionReason(
        code="test_reason",
        message="less severe reason",
        severity=1.0,
    )
    more_severe_reason = nvbench_compare.DecisionReason(
        code="test_reason",
        message="more severe reason",
        severity=2.0,
    )

    stats.record(nvbench_compare.ComparisonStatus.UNDECIDED, less_severe_reason)
    stats.record(nvbench_compare.ComparisonStatus.UNDECIDED, more_severe_reason)

    summary = stats.undecided_reasons["test_reason"]
    assert summary.count == 2
    assert summary.message == "more severe reason"


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


def test_compare_benches_reports_regression_when_robust_intervals_and_clock_confirm(
    monkeypatch, nvbench_compare
):
    run_data = make_comparison_run_data(nvbench_compare)

    ref_state = make_state(nvbench_compare, "state", mean="1.0", noise="0.01")
    ref_state["summaries"].extend(
        [
            make_summary(nvbench_compare, "GPU_TIME_MIN_TAG", "0.9"),
            make_summary(nvbench_compare, "GPU_TIME_Q1_TAG", "0.95"),
            make_summary(nvbench_compare, "GPU_TIME_MEDIAN_TAG", "1.0"),
            make_summary(nvbench_compare, "GPU_TIME_Q3_TAG", "1.05"),
            make_summary(nvbench_compare, "GPU_TIME_IR_RELATIVE_TAG", "0.01"),
            make_summary(nvbench_compare, "GPU_SM_CLOCK_RATE_MEAN_TAG", "100.0"),
        ]
    )
    cmp_state = make_state(nvbench_compare, "state", mean="1.0", noise="0.01")
    cmp_state["summaries"].extend(
        [
            make_summary(nvbench_compare, "GPU_TIME_MIN_TAG", "1.15"),
            make_summary(nvbench_compare, "GPU_TIME_Q1_TAG", "1.18"),
            make_summary(nvbench_compare, "GPU_TIME_MEDIAN_TAG", "1.2"),
            make_summary(nvbench_compare, "GPU_TIME_Q3_TAG", "1.25"),
            make_summary(nvbench_compare, "GPU_TIME_IR_RELATIVE_TAG", "0.01"),
            make_summary(nvbench_compare, "GPU_SM_CLOCK_RATE_MEAN_TAG", "100.0"),
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
    assert run_data.stats.undecided_count == 0
    assert run_data.stats.unknown_count == 0


def test_compare_benches_accepts_custom_comparison_thresholds(
    monkeypatch, nvbench_compare
):
    run_data = make_comparison_run_data(nvbench_compare)

    ref_state = make_state(nvbench_compare, "state", mean="1.0", noise="0.01")
    ref_state["summaries"].extend(
        [
            make_summary(nvbench_compare, "GPU_TIME_MIN_TAG", "0.99"),
            make_summary(nvbench_compare, "GPU_TIME_Q1_TAG", "0.995"),
            make_summary(nvbench_compare, "GPU_TIME_MEDIAN_TAG", "1.0"),
            make_summary(nvbench_compare, "GPU_TIME_Q3_TAG", "1.01"),
            make_summary(nvbench_compare, "GPU_TIME_IR_RELATIVE_TAG", "0.01"),
        ]
    )
    cmp_state = make_state(nvbench_compare, "state", mean="1.01", noise="0.01")
    cmp_state["summaries"].extend(
        [
            make_summary(nvbench_compare, "GPU_TIME_MIN_TAG", "1.0"),
            make_summary(nvbench_compare, "GPU_TIME_Q1_TAG", "1.005"),
            make_summary(nvbench_compare, "GPU_TIME_MEDIAN_TAG", "1.01"),
            make_summary(nvbench_compare, "GPU_TIME_Q3_TAG", "1.02"),
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
        comparison_thresholds=nvbench_compare.ComparisonThresholds(
            same_center_relative=0.02
        ),
    )

    assert run_data.stats.config_count == 1
    assert run_data.stats.pass_count == 1
    assert run_data.stats.undecided_count == 0


def test_compare_benches_marks_unavailable_noise_undecided(
    monkeypatch, nvbench_compare
):
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
    assert run_data.stats.undecided_count == 2
    assert run_data.stats.unknown_count == 0


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
    assert run_data.stats.pass_count == 0
    assert run_data.stats.improvement_count == 0
    assert run_data.stats.regression_count == 0
    assert run_data.stats.undecided_count == 2
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
    assert run_data.stats.pass_count == 0
    assert run_data.stats.improvement_count == 0
    assert run_data.stats.regression_count == 0
    assert run_data.stats.undecided_count == 1
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
    assert run_data.stats.pass_count == 0
    assert run_data.stats.improvement_count == 0
    assert run_data.stats.regression_count == 0
    assert run_data.stats.undecided_count == 3
    assert run_data.stats.unknown_count == 0


def test_main_returns_success_exit_code_when_regressions_are_detected(
    monkeypatch, capsys, nvbench_compare
):
    devices = [{"id": 0, "name": "Test GPU"}]
    ref_state = make_state(nvbench_compare, "state", mean="1.0")
    ref_state["summaries"].extend(
        [
            make_summary(nvbench_compare, "GPU_TIME_MIN_TAG", "0.9"),
            make_summary(nvbench_compare, "GPU_TIME_Q1_TAG", "0.95"),
            make_summary(nvbench_compare, "GPU_TIME_MEDIAN_TAG", "1.0"),
            make_summary(nvbench_compare, "GPU_TIME_Q3_TAG", "1.05"),
            make_summary(nvbench_compare, "GPU_SM_CLOCK_RATE_MEAN_TAG", "100.0"),
        ]
    )
    cmp_state = make_state(nvbench_compare, "state", mean="1.2")
    cmp_state["summaries"].extend(
        [
            make_summary(nvbench_compare, "GPU_TIME_MIN_TAG", "1.15"),
            make_summary(nvbench_compare, "GPU_TIME_Q1_TAG", "1.18"),
            make_summary(nvbench_compare, "GPU_TIME_MEDIAN_TAG", "1.2"),
            make_summary(nvbench_compare, "GPU_TIME_Q3_TAG", "1.25"),
            make_summary(nvbench_compare, "GPU_SM_CLOCK_RATE_MEAN_TAG", "100.0"),
        ]
    )
    ref_root = {
        "devices": devices,
        "benchmarks": [make_benchmark([ref_state])],
    }
    cmp_root = {
        "devices": devices,
        "benchmarks": [make_benchmark([cmp_state])],
    }

    def read_file(path):
        return ref_root if path == "ref.json" else cmp_root

    monkeypatch.setattr(nvbench_compare.reader, "read_file", read_file)
    monkeypatch.setattr(sys, "argv", ["nvbench_compare", "ref.json", "cmp.json"])

    assert nvbench_compare.main() == 0
    assert "Regression  (clear timing gap, %Diff > 0): 1" in capsys.readouterr().out


def test_main_prints_undecided_reason_summary(monkeypatch, capsys, nvbench_compare):
    devices = [{"id": 0, "name": "Test GPU"}]
    ref_root = {
        "devices": devices,
        "benchmarks": [
            make_benchmark([make_state(nvbench_compare, "state", noise="0.05")])
        ],
    }
    cmp_root = {
        "devices": devices,
        "benchmarks": [
            make_benchmark(
                [make_state(nvbench_compare, "state", mean="1.01", noise="0.05")]
            )
        ],
    }

    def read_file(path):
        return ref_root if path == "ref.json" else cmp_root

    monkeypatch.setattr(nvbench_compare.reader, "read_file", read_file)
    monkeypatch.setattr(sys, "argv", ["nvbench_compare", "ref.json", "cmp.json"])

    assert nvbench_compare.main() == 0
    output = capsys.readouterr().out
    assert "Undecided   (comparison requires more evidence): 1" in output
    assert "noise_too_high: 1" in output


def test_get_comparison_thresholds_returns_named_presets(nvbench_compare):
    default = nvbench_compare.get_comparison_thresholds("default")
    strict = nvbench_compare.get_comparison_thresholds("strict")
    permissive = nvbench_compare.get_comparison_thresholds("permissive")

    assert default == nvbench_compare.ComparisonThresholds(
        **nvbench_compare.COMPARISON_THRESHOLD_PRESET_VALUES["default"]
    )
    assert strict.clear_gap_relative > default.clear_gap_relative
    assert strict.same_center_relative < default.same_center_relative
    assert strict.bulk_same_sample_coverage > default.bulk_same_sample_coverage
    assert permissive.clear_gap_relative < default.clear_gap_relative
    assert permissive.same_center_relative > default.same_center_relative
    assert permissive.bulk_same_support_coverage < default.bulk_same_support_coverage


def test_compare_benches_defaults_to_interval_display(monkeypatch, nvbench_compare):
    run_data = make_comparison_run_data(nvbench_compare)
    captured = {}

    def fake_tabulate(rows, headers, *args, **kwargs):
        captured["rows"] = rows
        captured["headers"] = headers
        return ""

    monkeypatch.setattr(nvbench_compare.tabulate, "tabulate", fake_tabulate)

    ref_benches = [make_benchmark([make_state(nvbench_compare, "state", mean="1.0")])]
    cmp_benches = [make_benchmark([make_state(nvbench_compare, "state", mean="1.01")])]

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

    assert captured["headers"][-3:] == ["Ref", "Cmp", "Status"]
    row = captured["rows"][0]
    assert row[-3].startswith("1.000 s")
    assert row[-2].startswith("1.010 s")


def test_compare_benches_legacy_display_uses_scalar_diff(monkeypatch, nvbench_compare):
    run_data = make_comparison_run_data(nvbench_compare)
    captured = {}

    def fake_tabulate(rows, headers, *args, **kwargs):
        captured["rows"] = rows
        captured["headers"] = headers
        return ""

    monkeypatch.setattr(nvbench_compare.tabulate, "tabulate", fake_tabulate)

    ref_benches = [make_benchmark([make_state(nvbench_compare, "state", mean="1.0")])]
    cmp_benches = [make_benchmark([make_state(nvbench_compare, "state", mean="1.01")])]

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
        display="legacy",
    )

    assert captured["headers"][-7:] == [
        "Ref Time",
        "Ref Noise",
        "Cmp Time",
        "Cmp Noise",
        "Diff",
        "%Diff",
        "Status",
    ]
    row = captured["rows"][0]
    assert row[-7] == "1.000 s"
    assert row[-5] == "1.010 s"
    assert row[-3] == "10.000 ms"
    assert row[-2] == "1.00%"


def test_compare_benches_explain_display_uses_explicit_intervals(
    monkeypatch, nvbench_compare
):
    run_data = make_comparison_run_data(nvbench_compare)
    captured = {}

    def fake_tabulate(rows, headers, *args, **kwargs):
        captured["rows"] = rows
        captured["headers"] = headers
        return ""

    monkeypatch.setattr(nvbench_compare.tabulate, "tabulate", fake_tabulate)

    ref_state = make_state(nvbench_compare, "state", mean="1.0")
    ref_state["summaries"].extend(
        [
            make_summary(nvbench_compare, "GPU_TIME_MIN_TAG", "1.0"),
            make_summary(nvbench_compare, "GPU_TIME_Q1_TAG", "1.01"),
            make_summary(nvbench_compare, "GPU_TIME_MEDIAN_TAG", "1.02"),
            make_summary(nvbench_compare, "GPU_TIME_Q3_TAG", "1.03"),
            make_summary(nvbench_compare, "GPU_SM_CLOCK_RATE_MEAN_TAG", "100.0"),
        ]
    )
    cmp_state = make_state(nvbench_compare, "state", mean="1.01")
    cmp_state["summaries"].extend(
        [
            make_summary(nvbench_compare, "GPU_TIME_MIN_TAG", "1.01"),
            make_summary(nvbench_compare, "GPU_TIME_Q1_TAG", "1.02"),
            make_summary(nvbench_compare, "GPU_TIME_MEDIAN_TAG", "1.03"),
            make_summary(nvbench_compare, "GPU_TIME_Q3_TAG", "1.04"),
            make_summary(nvbench_compare, "GPU_SM_CLOCK_RATE_MEAN_TAG", "100.0"),
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
        display="explain",
    )

    assert captured["headers"][-6:] == [
        "Ref [L | C | H]",
        "Cmp [L | C | H]",
        "Ref Noise",
        "Cmp Noise",
        "Reason",
        "Status",
    ]
    row = captured["rows"][0]
    assert row[-6] == "1.0[00 | 20 | 30] s"
    assert row[-5] == "1.0[10 | 30 | 40] s"


def test_main_passes_selected_preset_to_compare_benches(monkeypatch, nvbench_compare):
    devices = [{"id": 0, "name": "Test GPU"}]
    root = {
        "devices": devices,
        "benchmarks": [],
    }
    captured = {}

    monkeypatch.setattr(nvbench_compare.reader, "read_file", lambda _: root)

    def fake_compare_benches(*args, **kwargs):
        captured["comparison_thresholds"] = kwargs["comparison_thresholds"]
        captured["display"] = kwargs["display"]

    monkeypatch.setattr(nvbench_compare, "compare_benches", fake_compare_benches)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nvbench_compare",
            "--preset",
            "strict",
            "--display",
            "explain",
            "ref.json",
            "cmp.json",
        ],
    )

    assert nvbench_compare.main() == 0
    assert captured[
        "comparison_thresholds"
    ] == nvbench_compare.get_comparison_thresholds("strict")
    assert captured["display"] == "explain"
