# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib.util
import math
import sys
import types
from dataclasses import replace
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
                LIGHTBLACK_EX="",
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
    monkeypatch.setitem(sys.modules, spec.name, module)
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


def make_raw_binary_size_summary(nvbench_compare, value):
    return {
        "tag": nvbench_compare.SAMPLE_TIMES_TAG,
        "data": [{"name": "size", "type": "int64", "value": value}],
    }


def make_reader_for_roots(ref_root, cmp_root):
    def read_file(path):
        if path == "ref.json":
            return ref_root
        if path == "cmp.json":
            return cmp_root
        raise AssertionError(f"unexpected path: {path!r}")

    return read_file


def capture_tabulate_calls(monkeypatch, nvbench_compare):
    calls = []

    def fake_tabulate(rows, headers, *args, **kwargs):
        calls.append({"rows": rows, "headers": headers})
        return ""

    monkeypatch.setattr(nvbench_compare.tabulate, "tabulate", fake_tabulate)
    return calls


def find_tabulate_call(calls, expected_header_suffix):
    return next(
        call
        for call in calls
        if call["headers"][-len(expected_header_suffix) :] == expected_header_suffix
    )


INTERVAL_DISPLAY_HEADERS = ["Ref", "Cmp", "Change", "Status"]
LEGACY_DISPLAY_HEADERS = [
    "Ref Time",
    "Ref Noise",
    "Cmp Time",
    "Cmp Noise",
    "Diff",
    "%Diff",
    "Status",
]
EXPLAIN_DISPLAY_HEADERS = [
    "Ref [Lo | Ce | Hi]",
    "Cmp [Lo | Ce | Hi]",
    "Ref Noise",
    "Cmp Noise",
    "Reason",
    "Change",
    "Status",
]


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
    interquartile_range_relative=0.01,
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


def test_compare_benches_matches_duplicate_states_by_axis_values(
    monkeypatch, nvbench_compare
):
    run_data = make_comparison_run_data(nvbench_compare)
    observed_pairs = []

    def fake_compare_gpu_timings(ref_timing, cmp_timing, comparison_thresholds=None):
        del comparison_thresholds
        observed_pairs.append((ref_timing.mean, cmp_timing.mean))
        ref_estimate = nvbench_compare.TimeEstimate(
            center=ref_timing.mean, relative_dispersion=ref_timing.stdev_relative
        )
        cmp_estimate = nvbench_compare.TimeEstimate(
            center=cmp_timing.mean, relative_dispersion=cmp_timing.stdev_relative
        )
        return nvbench_compare.SummaryComparison(
            ref_interval=None,
            cmp_interval=None,
            ref_estimate=ref_estimate,
            cmp_estimate=cmp_estimate,
            ref_time=ref_timing.mean,
            cmp_time=cmp_timing.mean,
            ref_noise=ref_timing.stdev_relative,
            cmp_noise=cmp_timing.stdev_relative,
            diff=cmp_timing.mean - ref_timing.mean,
            frac_diff=0.0,
            diff_interval=None,
            frac_diff_interval=None,
            max_noise=0.0,
            status=nvbench_compare.ComparisonStatus.SAME,
            reason=nvbench_compare.DecisionReason("test", "test"),
        )

    monkeypatch.setattr(
        nvbench_compare, "compare_gpu_timings", fake_compare_gpu_timings
    )

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
        threshold=1.0,
        plot_along=None,
        plot=False,
        dark=False,
        filter_plan=make_filter_plan(nvbench_compare),
        no_color=True,
    )

    assert observed_pairs == [(2.0, 2.0), (1.0, 1.0)]
    assert run_data.stats.config_count == 2


def test_compare_benches_counts_non_finite_centers_as_unknown(
    monkeypatch, nvbench_compare
):
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

    assert run_data.stats.config_count == 3
    assert run_data.stats.pass_count == 0
    assert run_data.stats.improvement_count == 0
    assert run_data.stats.regression_count == 0
    assert run_data.stats.undecided_count == 1
    assert run_data.stats.unknown_count == 2


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


@pytest.mark.parametrize("value", [3, "3"])
def test_extract_binary_size_accepts_integral_values(value, nvbench_compare):
    assert (
        nvbench_compare.extract_binary_size(
            make_raw_binary_size_summary(nvbench_compare, value)
        )
        == 3
    )


@pytest.mark.parametrize("value", [True, False, 3.0, 3.5, "3.5"])
def test_extract_binary_size_rejects_non_integral_values(value, nvbench_compare):
    with pytest.raises(ValueError, match="is not an int64"):
        nvbench_compare.extract_binary_size(
            make_raw_binary_size_summary(nvbench_compare, value)
        )


def test_compare_benches_collects_bulk_debug_rows(tmp_path, nvbench_compare):
    run_data = make_comparison_run_data(nvbench_compare)
    ref_samples_file = tmp_path / "ref-samples.bin"
    ref_freqs_file = tmp_path / "ref-freqs.bin"
    cmp_samples_file = tmp_path / "cmp-samples.bin"
    cmp_freqs_file = tmp_path / "cmp-freqs.bin"
    np.array([1.0, 1.0], dtype="<f4").tofile(ref_samples_file)
    np.array([100.0, 100.0], dtype="<f4").tofile(ref_freqs_file)
    np.array([1.0, 1.0], dtype="<f4").tofile(cmp_samples_file)
    np.array([100.0, 100.0], dtype="<f4").tofile(cmp_freqs_file)

    ref_state = make_state(nvbench_compare, "state", mean="1.0")
    ref_state["summaries"].extend(
        [
            make_binary_summary(
                nvbench_compare, "SAMPLE_TIMES_TAG", str(ref_samples_file), 2
            ),
            make_binary_summary(
                nvbench_compare, "SAMPLE_FREQUENCIES_TAG", str(ref_freqs_file), 2
            ),
        ]
    )
    cmp_state = make_state(nvbench_compare, "state", mean="1.01")
    cmp_state["summaries"].extend(
        [
            make_binary_summary(
                nvbench_compare, "SAMPLE_TIMES_TAG", str(cmp_samples_file), 2
            ),
            make_binary_summary(
                nvbench_compare, "SAMPLE_FREQUENCIES_TAG", str(cmp_freqs_file), 2
            ),
        ]
    )
    bulk_debug_rows = []

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
        ref_json_dir=str(tmp_path),
        cmp_json_dir=str(tmp_path),
        ref_json_path="ref.json",
        cmp_json_path="cmp.json",
        bulk_debug_rows=bulk_debug_rows,
    )

    assert len(bulk_debug_rows) == 1
    row = bulk_debug_rows[0]
    assert row["row_index"] == 0
    assert row["table_row_index"] == 0
    assert row["benchmark"] == "bench"
    assert row["reference_json"] == "ref.json"
    assert row["compare_json"] == "cmp.json"
    assert row["status"] == nvbench_compare.ComparisonStatus.SAME.value
    assert row["occurrence"] == 0
    assert row["occurrence_count"] == 1
    assert row["reference_sample_filename"] == str(ref_samples_file)
    assert row["reference_sample_count"] == 2
    assert row["reference_frequency_filename"] == str(ref_freqs_file)
    assert row["compare_sample_filename"] == str(cmp_samples_file)
    assert row["compare_frequency_filename"] == str(cmp_freqs_file)


def test_format_bulk_debug_python_loads_arrays(tmp_path, nvbench_compare):
    samples_file = tmp_path / "samples.bin"
    np.array([1.0, 2.0], dtype="<f4").tofile(samples_file)
    script = nvbench_compare.format_bulk_debug_python(
        [
            {
                "reference_sample_filename": str(samples_file),
                "reference_sample_count": 2,
                "reference_frequency_filename": None,
                "reference_frequency_count": None,
                "compare_sample_filename": None,
                "compare_sample_count": None,
                "compare_frequency_filename": None,
                "compare_frequency_count": None,
            }
        ]
    )
    namespace = {}

    assert script.startswith("# NVB-BULK-BEGIN\n")
    assert script.endswith("# NVB-BULK-END\n")
    exec(script, namespace)

    arrays = namespace["load_bulk_data"](namespace["bulk_rows"][0])
    assert list(arrays["reference_samples"]) == pytest.approx([1.0, 2.0])
    assert arrays["reference_frequencies"] is None


def test_format_bulk_debug_python_handles_nonfinite_values(nvbench_compare):
    script = nvbench_compare.format_bulk_debug_python(
        [
            {
                "reference_time": math.nan,
                "compare_time": math.inf,
                "fractional_difference": -math.inf,
            }
        ]
    )
    namespace = {}

    assert 'nan = float("nan")' in script
    assert 'inf = float("inf")' in script
    assert "'reference_time': nan" in script
    assert "'compare_time': inf" in script
    assert "'fractional_difference': -inf" in script
    exec(script, namespace)

    row = namespace["bulk_rows"][0]
    assert math.isnan(row["reference_time"])
    assert row["compare_time"] == math.inf
    assert row["fractional_difference"] == -math.inf


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


def test_gpu_timing_data_accepts_legacy_ir_tags(nvbench_compare):
    timing = nvbench_compare.extract_gpu_timing_data(
        [
            make_summary(nvbench_compare, "LEGACY_GPU_TIME_IR_TAG", "0.5"),
            make_summary(nvbench_compare, "LEGACY_GPU_TIME_IR_RELATIVE_TAG", "0.25"),
        ],
    )

    assert timing.interquartile_range == pytest.approx(0.5)
    assert timing.interquartile_range_relative == pytest.approx(0.25)


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


def test_bulk_file_resolution_does_not_fall_back_to_cwd(
    monkeypatch, tmp_path, nvbench_compare
):
    json_dir = tmp_path / "json"
    cwd = tmp_path / "cwd"
    json_dir.mkdir()
    cwd.mkdir()
    np.array([123.0], dtype="<f4").tofile(cwd / "samples.bin")
    monkeypatch.chdir(cwd)

    assert nvbench_compare.resolve_binary_filename(str(json_dir), "samples.bin") == str(
        json_dir / "samples.bin"
    )

    timing = nvbench_compare.extract_gpu_timing_data(
        [
            make_binary_summary(nvbench_compare, "SAMPLE_TIMES_TAG", "samples.bin", 1),
        ],
        str(json_dir),
    )

    with pytest.warns(RuntimeWarning, match="failed to read"):
        assert timing.samples is None


def test_compare_gpu_timings_classifies_common_cases(tmp_path, nvbench_compare):
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

    partial_robust = nvbench_compare.compare_gpu_timings(
        make_gpu_timing_data(
            nvbench_compare,
            minimum=0.8,
            maximum=1.2,
            mean=1.0,
            stdev=0.1,
            stdev_relative=0.1,
            median=10.0,
            interquartile_range_relative=0.01,
        ),
        make_gpu_timing_data(
            nvbench_compare,
            minimum=0.85,
            maximum=1.25,
            mean=1.05,
            stdev=0.1,
            stdev_relative=0.1,
            median=11.0,
            interquartile_range_relative=0.01,
        ),
    )
    assert partial_robust is not None
    assert partial_robust.ref_time == pytest.approx(1.0)
    assert partial_robust.cmp_time == pytest.approx(1.05)
    assert partial_robust.ref_interval.center == pytest.approx(1.0)
    assert partial_robust.cmp_interval.center == pytest.approx(1.05)

    mixed_summary_families = nvbench_compare.compare_gpu_timings(
        make_gpu_timing_data(
            nvbench_compare,
            minimum=0.8,
            maximum=1.2,
            mean=1.0,
            stdev=0.1,
            stdev_relative=0.1,
            first_quartile=9.0,
            median=10.0,
            third_quartile=11.0,
            interquartile_range_relative=0.01,
        ),
        make_gpu_timing_data(
            nvbench_compare,
            minimum=0.85,
            maximum=1.25,
            mean=1.05,
            stdev=0.1,
            stdev_relative=0.1,
        ),
    )
    assert mixed_summary_families is not None
    assert mixed_summary_families.ref_time == pytest.approx(1.0)
    assert mixed_summary_families.cmp_time == pytest.approx(1.05)
    assert mixed_summary_families.ref_interval.center == pytest.approx(1.0)
    assert mixed_summary_families.cmp_interval.center == pytest.approx(1.05)

    mixed_robust_summary_and_bulk = nvbench_compare.compare_gpu_timings(
        make_gpu_timing_data(
            nvbench_compare,
            minimum=0.8,
            maximum=1.2,
            mean=1.0,
            stdev=0.1,
            stdev_relative=0.1,
            first_quartile=9.0,
            median=10.0,
            third_quartile=11.0,
            interquartile_range_relative=0.01,
            sample_values=[1.0, 2.0, 3.0],
        ),
        make_gpu_timing_data(
            nvbench_compare,
            minimum=98.0,
            maximum=102.0,
            mean=100.0,
            stdev=1.0,
            stdev_relative=0.01,
            sample_values=[4.0, 5.0, 6.0, 100.0],
        ),
    )
    assert mixed_robust_summary_and_bulk is not None
    assert mixed_robust_summary_and_bulk.ref_time == pytest.approx(10.0)
    assert mixed_robust_summary_and_bulk.cmp_time == pytest.approx(6.0)
    assert mixed_robust_summary_and_bulk.ref_interval.center == pytest.approx(10.0)
    assert mixed_robust_summary_and_bulk.cmp_interval.center == pytest.approx(6.0)
    assert mixed_robust_summary_and_bulk.cmp_interval.upper == pytest.approx(6.0)

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

    deterministic_same = nvbench_compare.compare_gpu_timings(
        make_gpu_timing_data(
            nvbench_compare,
            minimum=1.0,
            maximum=1.0,
            mean=1.0,
            stdev=0.0,
            stdev_relative=0.0,
        ),
        make_gpu_timing_data(
            nvbench_compare,
            minimum=1.0,
            maximum=1.0,
            mean=1.0,
            stdev=0.0,
            stdev_relative=0.0,
        ),
    )
    assert deterministic_same is not None
    assert deterministic_same.status == nvbench_compare.ComparisonStatus.SAME
    assert deterministic_same.reason.code == "same_without_clock_rate"
    assert deterministic_same.diff_interval == pytest.approx((0.0, 0.0))
    assert deterministic_same.frac_diff_interval == pytest.approx((0.0, 0.0))

    negative_noise = nvbench_compare.compare_gpu_timings(
        ref_interval_timing,
        make_gpu_timing_data(
            nvbench_compare,
            minimum=1.02,
            first_quartile=1.1,
            median=1.204,
            third_quartile=1.28,
            mean=1.204,
            interquartile_range_relative=-0.01,
            sm_clock_rate_mean=100.0,
        ),
    )
    assert negative_noise is not None
    assert negative_noise.status == nvbench_compare.ComparisonStatus.UNDECIDED
    assert negative_noise.max_noise is None
    assert negative_noise.reason.code == "noise_unavailable"

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

    missing_source = nvbench_compare.Float32BinarySource(
        count=4,
        filename="missing.bin",
        json_dir=str(tmp_path),
        description="test sample",
    )
    missing_bulk_files = nvbench_compare.compare_gpu_timings(
        replace(
            ref_interval_timing,
            sample_source=missing_source,
            frequency_source=missing_source,
        ),
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
    assert missing_bulk_files is not None
    assert missing_bulk_files.status == nvbench_compare.ComparisonStatus.FAST
    assert missing_bulk_files.reason.code == "clear_gap_confirmed_by_summary_cycles"

    unusable_bulk_cycles = nvbench_compare.compare_gpu_timings(
        make_gpu_timing_data(
            nvbench_compare,
            minimum=1.0,
            first_quartile=1.1,
            median=1.2,
            third_quartile=1.3,
            mean=1.2,
            stdev_relative=0.05,
            sm_clock_rate_mean=100.0,
            sample_values=[0.0, 1.1, 1.2, 1.3],
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
            sm_clock_rate_mean=100.0,
            sample_values=[0.8, 0.85, 0.9, 0.95],
            frequency_values=[100.0] * 4,
        ),
    )
    assert unusable_bulk_cycles is not None
    assert unusable_bulk_cycles.status == nvbench_compare.ComparisonStatus.UNDECIDED
    assert unusable_bulk_cycles.reason.code == "bulk_cycle_data_unusable"

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
    assert nvbench_compare.format_duration(None) == "n/a"
    assert nvbench_compare.format_duration(math.nan) == "n/a"
    assert nvbench_compare.format_duration(math.inf) == "n/a"
    assert nvbench_compare.format_duration(-1.0) == "n/a"
    assert nvbench_compare.format_duration(0.0) == "n/a"
    assert (
        nvbench_compare.format_duration(-1.0, allow_negative=True) == "-1000000.000 us"
    )
    assert nvbench_compare.format_duration(0.0, allow_zero=True) == "0.000 us"
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


def test_format_change_only_reports_fast_and_slow_rows(nvbench_compare):
    fast = types.SimpleNamespace(
        status=nvbench_compare.ComparisonStatus.FAST,
        frac_diff_interval=(-0.3, -0.05),
    )
    slow = types.SimpleNamespace(
        status=nvbench_compare.ComparisonStatus.SLOW,
        frac_diff_interval=(0.07, 0.55),
    )
    same = types.SimpleNamespace(
        status=nvbench_compare.ComparisonStatus.SAME,
        frac_diff_interval=(-0.01, 0.01),
    )
    undecided = types.SimpleNamespace(
        status=nvbench_compare.ComparisonStatus.UNDECIDED,
        frac_diff_interval=(-0.01, 0.01),
    )

    assert nvbench_compare.format_change(fast) == "<= -5.0%"
    assert nvbench_compare.format_change(slow) == ">= +7.0%"
    assert nvbench_compare.format_change(same) == ""
    assert nvbench_compare.format_change(undecided) == ""


def test_ambiguous_status_uses_shrug_marker(nvbench_compare):
    assert (
        nvbench_compare.colorize_comparison_status(
            nvbench_compare.ComparisonStatus.UNDECIDED, no_color=True
        )
        == "\U0001f937 AMBG"
    )


def test_format_timing_with_interval(nvbench_compare):
    interval = nvbench_compare.TimingInterval(
        lower=0.002237, upper=0.002389, center=0.0023
    )
    assert (
        nvbench_compare.format_timing_with_interval(0.0023, interval)
        == "2.300 ms [-63, +89] us"
    )

    interval = nvbench_compare.TimingInterval(
        lower=19.380e-6, upper=20.508e-6, center=19.944e-6
    )
    assert (
        nvbench_compare.format_timing_with_interval(19.944e-6, interval)
        == "19.944 [-0.564, +0.564] us"
    )


def test_format_timing_with_explicit_interval(nvbench_compare):
    interval = nvbench_compare.TimingInterval(
        lower=0.001434, upper=0.001458, center=0.001446
    )
    assert (
        nvbench_compare.format_timing_with_explicit_interval(0.001446, interval)
        == "1.4[34 | 46 | 58] ms"
    )

    interval = nvbench_compare.TimingInterval(
        lower=18.400e-6, upper=19.464e-6, center=18.736e-6
    )
    assert (
        nvbench_compare.format_timing_with_explicit_interval(18.736e-6, interval)
        == "[18.400 | 18.736 | 19.464] us"
    )

    interval = nvbench_compare.TimingInterval(
        lower=19.380e-6, upper=20.508e-6, center=19.944e-6
    )
    assert (
        nvbench_compare.format_timing_with_explicit_interval(19.944e-6, interval)
        == "[19.380 | 19.944 | 20.508] us"
    )

    interval = nvbench_compare.TimingInterval(
        lower=99.094e-6, upper=100.882e-6, center=99.988e-6
    )
    assert (
        nvbench_compare.format_timing_with_explicit_interval(99.988e-6, interval)
        == "[ 99.094 |  99.988 | 100.882] us"
    )


def test_align_explain_interval_columns_pads_values_across_rows(nvbench_compare):
    rows = [["", ""], ["", ""]]
    comparisons = [
        types.SimpleNamespace(
            ref_time=19.944e-6,
            ref_interval=nvbench_compare.TimingInterval(
                lower=19.380e-6, center=19.944e-6, upper=20.508e-6
            ),
            cmp_time=97.712e-6,
            cmp_interval=nvbench_compare.TimingInterval(
                lower=96.849e-6, center=97.712e-6, upper=98.574e-6
            ),
        ),
        types.SimpleNamespace(
            ref_time=103.466e-6,
            ref_interval=nvbench_compare.TimingInterval(
                lower=102.739e-6, center=103.466e-6, upper=104.193e-6
            ),
            cmp_time=101.868e-6,
            cmp_interval=nvbench_compare.TimingInterval(
                lower=100.916e-6, center=101.868e-6, upper=102.819e-6
            ),
        ),
    ]

    nvbench_compare.align_explain_interval_columns(rows, comparisons, axis_count=0)

    assert rows[0][0] == "[ 19.380 |  19.944 |  20.508] us"
    assert rows[1][0] == "[102.739 | 103.466 | 104.193] us"
    assert rows[0][1] == "[ 96.849 |  97.712 |  98.574] us"
    assert rows[1][1] == "[100.916 | 101.868 | 102.819] us"


def test_align_timing_interval_columns_reserves_missing_interval_slot(nvbench_compare):
    rows = [["", ""], ["", ""]]
    comparisons = [
        types.SimpleNamespace(
            ref_time=19.944e-6,
            ref_interval=nvbench_compare.TimingInterval(
                lower=19.380e-6, center=19.944e-6, upper=20.508e-6
            ),
            cmp_time=18.736e-6,
            cmp_interval=nvbench_compare.TimingInterval(
                lower=18.400e-6, center=18.736e-6, upper=19.464e-6
            ),
        ),
        types.SimpleNamespace(
            ref_time=20.390e-6,
            ref_interval=nvbench_compare.TimingInterval(
                lower=19.659e-6, center=20.390e-6, upper=21.121e-6
            ),
            cmp_time=20.480e-6,
            cmp_interval=None,
        ),
    ]

    nvbench_compare.align_timing_interval_columns(rows, comparisons, axis_count=0)

    cmp_interval_slot = len("[-0.336, +0.728]")
    assert rows[0][1] == "18.736 [-0.336, +0.728] us"
    assert rows[1][1] == f"20.480 {' ' * cmp_interval_slot} us"


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
    sample_threshold = (
        nvbench_compare.get_default_thresholds().bulk_same_sample_coverage * 100.0
    )
    assert (
        f"sample: min(ref=0.0%, cmp=0.0%) >= {sample_threshold:0.1f}%"
        in comparison.reason.message
    )
    assert "support: min(ref=0.0%, cmp=0.0%) >= 80.0%" in comparison.reason.message
    assert f"{sample_threshold:0.1f}%" in comparison.reason.message
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
        thresholds=nvbench_compare.get_default_thresholds(),
    )

    assert decision.status == nvbench_compare.ComparisonStatus.UNDECIDED
    assert decision.reason.code == "bulk_time_support_mismatch"
    assert "sample: min(ref=3.8%, cmp=100.0%) >= 97.0%" in decision.reason.message
    assert "support: min(ref=80.0%, cmp=100.0%) >= 80.0%" in decision.reason.message


def test_bulk_same_filters_rare_values_from_support_coverage(nvbench_compare):
    ref_values = [1.0] * 1000 + [1.02 + 0.01 * i for i in range(10)]
    cmp_values = [1.0]

    decision = nvbench_compare.compare_values_for_bulk_same(
        ref_values,
        cmp_values,
        label="time",
        thresholds=nvbench_compare.get_default_thresholds(),
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
        thresholds=replace(
            nvbench_compare.get_default_thresholds(),
            bulk_support_max_removed_sample_fraction=0.005,
        ),
    )

    assert decision.status == nvbench_compare.ComparisonStatus.UNDECIDED
    assert decision.reason.code == "bulk_time_support_mismatch"
    assert "sample: min(ref=99.0%, cmp=100.0%) >= 97.0%" in decision.reason.message
    assert "support: min(ref=9.1%, cmp=100.0%) >= 80.0%" in decision.reason.message


def test_bulk_same_retains_full_support_when_all_values_are_unique(nvbench_compare):
    coverages = nvbench_compare.compute_nearest_neighbor_coverages(
        [1.0, 1.02],
        [1.0],
        thresholds=replace(
            nvbench_compare.get_default_thresholds(),
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


def test_reason_legend_omits_trivial_aliases(nvbench_compare):
    reason_legend = {
        "bulk-same": nvbench_compare.DecisionReasonSummary(canonical_code="bulk_same"),
        "bt-sup-miss": nvbench_compare.DecisionReasonSummary(
            canonical_code="bulk_time_support_mismatch"
        ),
    }

    assert nvbench_compare.format_reason_legend_entries(reason_legend) == [
        "bt-sup-miss = bulk_time_support_mismatch"
    ]


@pytest.mark.parametrize(
    "ref_time, cmp_time, reason_code",
    [
        (None, 1.0, "timing_center_missing"),
        (1.0, None, "timing_center_missing"),
        (math.nan, 1.0, "timing_center_nonfinite"),
        (math.inf, 1.0, "timing_center_nonfinite"),
        (0.0, 1.0, "timing_center_nonpositive"),
        (-1.0, 1.0, "timing_center_nonpositive"),
    ],
)
def test_compare_gpu_timings_reports_unusable_centers_as_unknown(
    nvbench_compare, ref_time, cmp_time, reason_code
):
    comparison = nvbench_compare.compare_gpu_timings(
        make_gpu_timing_data(nvbench_compare, mean=ref_time),
        make_gpu_timing_data(nvbench_compare, mean=cmp_time),
    )

    assert comparison is not None
    assert comparison.status == nvbench_compare.ComparisonStatus.UNKNOWN
    assert comparison.reason.code == reason_code
    assert comparison.diff is None
    assert comparison.frac_diff is None


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
            make_summary(nvbench_compare, "GPU_TIME_IQR_RELATIVE_TAG", "0.01"),
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
            make_summary(nvbench_compare, "GPU_TIME_IQR_RELATIVE_TAG", "0.01"),
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
            make_summary(nvbench_compare, "GPU_TIME_IQR_RELATIVE_TAG", "0.01"),
        ]
    )
    cmp_state = make_state(nvbench_compare, "state", mean="1.01", noise="0.01")
    cmp_state["summaries"].extend(
        [
            make_summary(nvbench_compare, "GPU_TIME_MIN_TAG", "1.0"),
            make_summary(nvbench_compare, "GPU_TIME_Q1_TAG", "1.005"),
            make_summary(nvbench_compare, "GPU_TIME_MEDIAN_TAG", "1.01"),
            make_summary(nvbench_compare, "GPU_TIME_Q3_TAG", "1.02"),
            make_summary(nvbench_compare, "GPU_TIME_IQR_RELATIVE_TAG", "0.01"),
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
        comparison_thresholds=replace(
            nvbench_compare.get_default_thresholds(), same_center_relative=0.02
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
    plot_calls = []

    def fake_plot(x, y, shape, *args, **kwargs):
        plot_calls.append({"x": x, "y": y, "shape": shape, "label": kwargs["label"]})
        return [types.SimpleNamespace(get_color=lambda: "black")]

    monkeypatch.setattr(sys.modules["matplotlib.pyplot"], "plot", fake_plot)
    monkeypatch.setattr(
        nvbench_compare, "plot_comparison_entries", lambda *args, **kwargs: None
    )

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
        plot=True,
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
    assert [call["x"] for call in plot_calls] == [[1.0], [1.0]]
    assert [call["shape"] for call in plot_calls] == ["-", "--"]


def test_plot_along_rejects_non_numeric_axis_values(monkeypatch, nvbench_compare):
    run_data = make_comparison_run_data(nvbench_compare)

    ref_benches = [
        make_benchmark([make_state(nvbench_compare, "state", axis_value="F32")])
    ]
    cmp_benches = [
        make_benchmark([make_state(nvbench_compare, "state", axis_value="F32")])
    ]
    ref_benches[0]["axes"] = [{"name": "A", "type": "type", "flags": ""}]
    cmp_benches[0]["axes"] = [{"name": "A", "type": "type", "flags": ""}]

    with pytest.raises(
        ValueError,
        match="--plot-along requires numeric axis values; axis 'A' has value 'F32'",
    ):
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


@pytest.mark.parametrize("axis_value", ["0", "-1", "nan", "inf"])
def test_plot_along_rejects_non_positive_or_non_finite_axis_values(
    monkeypatch, nvbench_compare, axis_value
):
    run_data = make_comparison_run_data(nvbench_compare)

    ref_benches = [
        make_benchmark([make_state(nvbench_compare, "state", axis_value=axis_value)])
    ]
    cmp_benches = [
        make_benchmark([make_state(nvbench_compare, "state", axis_value=axis_value)])
    ]
    ref_benches[0]["axes"] = [{"name": "A", "type": "float64", "flags": ""}]
    cmp_benches[0]["axes"] = [{"name": "A", "type": "float64", "flags": ""}]

    with pytest.raises(
        ValueError,
        match="--plot-along requires positive finite axis values",
    ):
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


def test_main_warns_on_device_mismatch_with_explicit_device_filters(
    monkeypatch, capsys, nvbench_compare
):
    ref_root = {
        "devices": [{"id": 0, "name": "Reference GPU"}],
        "benchmarks": [],
    }
    cmp_root = {
        "devices": [{"id": 1, "name": "Compare GPU"}],
        "benchmarks": [],
    }

    monkeypatch.setattr(
        nvbench_compare.reader, "read_file", make_reader_for_roots(ref_root, cmp_root)
    )
    monkeypatch.setattr(
        nvbench_compare.jsondiff,
        "diff",
        lambda *args, **kwargs: {"name": ["Reference GPU", "Compare GPU"]},
    )
    monkeypatch.setattr(
        nvbench_compare, "compare_benches", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nvbench_compare",
            "--reference-devices",
            "0",
            "--compare-devices",
            "1",
            "ref.json",
            "cmp.json",
        ],
    )

    assert nvbench_compare.main() == 0
    output = capsys.readouterr().out
    assert "Device sections do not match" in output
    assert "Reference GPU" in output
    assert "Compare GPU" in output


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


def test_global_axis_filter_does_not_select_unmatched_benchmark(
    monkeypatch, nvbench_compare
):
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
            [("axis", "A=2"), ("benchmark", "bench1")],
        ),
        no_color=True,
    )

    assert run_data.stats.config_count == 1
    assert run_data.stats.pass_count == 0
    assert run_data.stats.improvement_count == 0
    assert run_data.stats.regression_count == 0
    assert run_data.stats.undecided_count == 1
    assert run_data.stats.unknown_count == 0


def test_global_axis_filter_applies_to_each_selected_benchmark(
    monkeypatch, nvbench_compare
):
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
            [("axis", "A=2"), ("benchmark", "bench1"), ("benchmark", "bench2")],
        ),
        no_color=True,
    )

    assert run_data.stats.config_count == 2
    assert run_data.stats.pass_count == 0
    assert run_data.stats.improvement_count == 0
    assert run_data.stats.regression_count == 0
    assert run_data.stats.undecided_count == 2
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
            make_summary(nvbench_compare, "GPU_TIME_IQR_RELATIVE_TAG", "0.01"),
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
            make_summary(nvbench_compare, "GPU_TIME_IQR_RELATIVE_TAG", "0.01"),
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

    monkeypatch.setattr(
        nvbench_compare.reader, "read_file", make_reader_for_roots(ref_root, cmp_root)
    )
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

    monkeypatch.setattr(
        nvbench_compare.reader, "read_file", make_reader_for_roots(ref_root, cmp_root)
    )
    monkeypatch.setattr(
        sys, "argv", ["nvbench_compare", "--display", "explain", "ref.json", "cmp.json"]
    )

    assert nvbench_compare.main() == 0
    output = capsys.readouterr().out
    assert "Ambiguous (comparison requires more evidence): 1" in output
    assert "noise_too_high: 1" in output
    assert "Reason legend: noise-high = noise_too_high" in output


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


def test_dump_comparison_config_uses_grouped_toml(nvbench_compare):
    config = nvbench_compare.dump_comparison_config(
        "default", nvbench_compare.get_comparison_thresholds("default")
    )

    assert "version = 1\n" in config
    assert '[preset]\nname = "default"\n' in config
    assert "[clear_gap]\nrelative = 0.005\n" in config
    assert "[same]\n" in config
    assert "[bulk]\n" in config
    assert "sample_coverage = 0.97\n" in config
    assert "[bulk.rare_support]\n" in config


def test_resolve_comparison_thresholds_applies_config_overrides(
    monkeypatch, nvbench_compare
):
    def read_config(_):
        return (
            "strict",
            {
                "bulk_same_sample_coverage": 0.93,
                "bulk_support_max_removed_sample_fraction": 0.02,
            },
        )

    monkeypatch.setattr(nvbench_compare, "read_comparison_config_file", read_config)

    preset, thresholds = nvbench_compare.resolve_comparison_thresholds(
        None, "settings.toml"
    )
    assert preset == "strict"
    assert thresholds.clear_gap_relative == pytest.approx(
        nvbench_compare.get_comparison_thresholds("strict").clear_gap_relative
    )
    assert thresholds.bulk_same_sample_coverage == pytest.approx(0.93)
    assert thresholds.bulk_support_max_removed_sample_fraction == pytest.approx(0.02)

    preset, thresholds = nvbench_compare.resolve_comparison_thresholds(
        "permissive", "settings.toml"
    )
    assert preset == "permissive"
    assert thresholds.clear_gap_relative == pytest.approx(
        nvbench_compare.get_comparison_thresholds("permissive").clear_gap_relative
    )
    assert thresholds.bulk_same_sample_coverage == pytest.approx(0.93)
    assert thresholds.bulk_support_max_removed_sample_fraction == pytest.approx(0.02)


def test_parse_comparison_config_data_validates_grouped_thresholds(nvbench_compare):
    preset, overrides = nvbench_compare.parse_comparison_config_data(
        {
            "version": 1,
            "preset": {"name": "strict"},
            "clear_gap": {"relative": 0.01},
            "same": {
                "center_relative": 0.002,
                "overlap_fraction": 0.75,
                "relative_dispersion_ceiling": 0.02,
            },
            "bulk": {
                "sample_coverage": 0.99,
                "support_coverage": 0.8,
                "rare_support": {
                    "sample_fraction": 0.001,
                    "max_removed_sample_fraction": 0.01,
                },
            },
        }
    )

    assert preset == "strict"
    assert overrides == {
        "clear_gap_relative": 0.01,
        "same_center_relative": 0.002,
        "same_overlap_fraction": 0.75,
        "same_relative_dispersion_ceiling": 0.02,
        "bulk_same_sample_coverage": 0.99,
        "bulk_same_support_coverage": 0.8,
        "bulk_support_rare_sample_fraction": 0.001,
        "bulk_support_max_removed_sample_fraction": 0.01,
    }


@pytest.mark.parametrize(
    "config_data, match",
    [
        ({}, "version"),
        ({"version": 2}, "unsupported"),
        ({"version": 1, "rare_support": {}}, "unknown top-level"),
        ({"version": 1, "bulk": {"unknown": 0.1}}, r"\[bulk\]"),
        ({"version": 1, "clear_gap": {"rare_support": {}}}, r"\[clear_gap\]"),
        ({"version": 1, "bulk": {"sample_coverage": 1.5}}, "<= 1"),
        ({"version": 1, "same": {"center_relative": "tight"}}, "finite number"),
        ({"version": 1, "preset": {"name": "aggressive"}}, "unknown comparison preset"),
    ],
)
def test_parse_comparison_config_data_rejects_invalid_config(
    nvbench_compare, config_data, match
):
    with pytest.raises(ValueError, match=match):
        nvbench_compare.parse_comparison_config_data(config_data)


@pytest.mark.skipif(
    importlib.util.find_spec("tomllib") is None
    and importlib.util.find_spec("tomli") is None,
    reason="TOML config support requires tomllib or tomli",
)
def test_read_comparison_config_file_parses_toml_with_available_parser(
    tmp_path, nvbench_compare
):
    config_path = tmp_path / "settings.toml"
    config_path.write_text(
        """
version = 1

[preset]
name = "strict"

[bulk]
sample_coverage = 0.93
""",
        encoding="utf-8",
    )

    preset, overrides = nvbench_compare.read_comparison_config_file(config_path)

    assert preset == "strict"
    assert overrides == {"bulk_same_sample_coverage": 0.93}


def test_main_dump_config_does_not_require_input_files(
    monkeypatch, capsys, nvbench_compare
):
    def read_file(_):
        raise AssertionError("dump-config should not read JSON files")

    monkeypatch.setattr(nvbench_compare.reader, "read_file", read_file)
    monkeypatch.setattr(
        sys,
        "argv",
        ["nvbench_compare", "--preset", "strict", "--dump-config"],
    )

    assert nvbench_compare.main() == 0
    output = capsys.readouterr().out
    assert 'name = "strict"' in output
    assert "[bulk.rare_support]" in output


def test_main_dump_config_merges_config_and_cli_preset(
    monkeypatch, capsys, nvbench_compare
):
    def read_config(_):
        return ("strict", {"bulk_same_sample_coverage": 0.93})

    monkeypatch.setattr(nvbench_compare, "read_comparison_config_file", read_config)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nvbench_compare",
            "--config",
            "settings.toml",
            "--preset",
            "permissive",
            "--dump-config",
        ],
    )

    assert nvbench_compare.main() == 0
    output = capsys.readouterr().out
    assert 'name = "permissive"' in output
    assert "relative = 0.0025" in output
    assert "sample_coverage = 0.93" in output


def test_main_rejects_unknown_options(monkeypatch, nvbench_compare):
    monkeypatch.setattr(
        sys,
        "argv",
        ["nvbench_compare", "--dispaly", "explain", "ref.json", "cmp.json"],
    )

    with pytest.raises(SystemExit) as exc_info:
        nvbench_compare.main()

    assert exc_info.value.code == 2


def test_main_reports_input_read_failures(monkeypatch, capsys, nvbench_compare):
    def read_file(_):
        raise OSError("cannot open file")

    monkeypatch.setattr(nvbench_compare.reader, "read_file", read_file)
    monkeypatch.setattr(sys, "argv", ["nvbench_compare", "ref.json", "cmp.json"])

    assert nvbench_compare.main() == 1
    output = capsys.readouterr().out
    assert "failed to read NVBench JSON file 'ref.json'" in output
    assert "cannot open file" in output


def test_main_reports_missing_required_root_keys(monkeypatch, capsys, nvbench_compare):
    monkeypatch.setattr(nvbench_compare.reader, "read_file", lambda _: {"devices": []})
    monkeypatch.setattr(sys, "argv", ["nvbench_compare", "ref.json", "cmp.json"])

    assert nvbench_compare.main() == 1
    output = capsys.readouterr().out
    assert "NVBench JSON file 'ref.json' is missing required root key(s)" in output
    assert "'benchmarks'" in output


def test_main_rejects_non_array_root_keys(monkeypatch, capsys, nvbench_compare):
    monkeypatch.setattr(
        nvbench_compare.reader,
        "read_file",
        lambda _: {"devices": {}, "benchmarks": []},
    )
    monkeypatch.setattr(sys, "argv", ["nvbench_compare", "ref.json", "cmp.json"])

    assert nvbench_compare.main() == 1
    output = capsys.readouterr().out
    assert "NVBench JSON file 'ref.json' root key 'devices' must be an array" in output


def test_main_rejects_non_object_root_array_entries(
    monkeypatch, capsys, nvbench_compare
):
    monkeypatch.setattr(
        nvbench_compare.reader,
        "read_file",
        lambda _: {"devices": [None], "benchmarks": []},
    )
    monkeypatch.setattr(sys, "argv", ["nvbench_compare", "ref.json", "cmp.json"])

    assert nvbench_compare.main() == 1
    output = capsys.readouterr().out
    assert (
        "NVBench JSON file 'ref.json' root key 'devices' entry 0 must be an object"
        in output
    )


def test_main_reports_invalid_device_entry_structure(
    monkeypatch, capsys, nvbench_compare
):
    monkeypatch.setattr(
        nvbench_compare.reader,
        "read_file",
        lambda _: {"devices": [{}], "benchmarks": []},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nvbench_compare",
            "--reference-devices",
            "0",
            "--compare-devices",
            "0",
            "ref.json",
            "cmp.json",
        ],
    )

    assert nvbench_compare.main() == 1
    output = capsys.readouterr().out
    assert "invalid NVBench JSON structure" in output
    assert "missing key 'id'" in output


def test_main_reports_invalid_benchmark_entry_structure(
    monkeypatch, capsys, nvbench_compare
):
    monkeypatch.setattr(
        nvbench_compare.reader,
        "read_file",
        lambda _: {"devices": [], "benchmarks": [{}]},
    )
    monkeypatch.setattr(sys, "argv", ["nvbench_compare", "ref.json", "cmp.json"])

    assert nvbench_compare.main() == 1
    output = capsys.readouterr().out
    assert "invalid NVBench JSON structure" in output
    assert "missing key 'name'" in output


def test_main_prints_bulk_debug_python_to_stdout(monkeypatch, capsys, nvbench_compare):
    devices = [{"id": 0, "name": "Test GPU"}]
    root = {
        "devices": devices,
        "benchmarks": [],
    }

    monkeypatch.setattr(nvbench_compare.reader, "read_file", lambda _: root)

    def fake_compare_benches(*args, **kwargs):
        kwargs["bulk_debug_rows"].append(
            {
                "row_index": 0,
                "status": "AMBG",
                "reference_sample_filename": None,
                "reference_sample_count": None,
                "reference_frequency_filename": None,
                "reference_frequency_count": None,
                "compare_sample_filename": None,
                "compare_sample_count": None,
                "compare_frequency_filename": None,
                "compare_frequency_count": None,
            }
        )

    monkeypatch.setattr(nvbench_compare, "compare_benches", fake_compare_benches)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nvbench_compare",
            "--bulk-debug-python",
            "STDOUT",
            "ref.json",
            "cmp.json",
        ],
    )

    assert nvbench_compare.main() == 0
    output = capsys.readouterr().out
    assert "# NVB-BULK-BEGIN" in output
    assert "bulk_rows = [" in output
    assert "'status': 'AMBG'" in output
    assert "def load_bulk_data(row):" in output
    assert "# NVB-BULK-END" in output


def test_compare_benches_counts_unusable_timing_as_unknown(
    monkeypatch, nvbench_compare
):
    run_data = make_comparison_run_data(nvbench_compare)
    tabulate_calls = capture_tabulate_calls(monkeypatch, nvbench_compare)

    ref_benches = [make_benchmark([make_state(nvbench_compare, "state", mean="nan")])]
    cmp_benches = [make_benchmark([make_state(nvbench_compare, "state", mean="1.0")])]

    nvbench_compare.compare_benches(
        run_data,
        ref_benches,
        cmp_benches,
        threshold=1.0,
        plot_along=None,
        plot=False,
        dark=False,
        filter_plan=make_filter_plan(nvbench_compare),
        no_color=True,
    )

    assert run_data.stats.config_count == 1
    assert run_data.stats.unknown_count == 1
    table = find_tabulate_call(tabulate_calls, INTERVAL_DISPLAY_HEADERS)
    row = table["rows"][0]
    assert row[-4] == "n/a"
    assert row[-3] == "1.000 s"
    assert row[-2] == ""
    assert row[-1] == "\U0001f7e1 ????"


def test_compare_benches_counts_skipped_state_as_unknown(monkeypatch, nvbench_compare):
    run_data = make_comparison_run_data(nvbench_compare)
    tabulate_calls = capture_tabulate_calls(monkeypatch, nvbench_compare)

    ref_state = make_state(nvbench_compare, "state")
    ref_state["summaries"] = None
    ref_state["is_skipped"] = True
    ref_state["skip_reason"] = "requested by benchmark"
    cmp_state = make_state(nvbench_compare, "state", mean="1.0")

    nvbench_compare.compare_benches(
        run_data,
        [make_benchmark([ref_state])],
        [make_benchmark([cmp_state])],
        threshold=1.0,
        plot_along=None,
        plot=False,
        dark=False,
        filter_plan=make_filter_plan(nvbench_compare),
        no_color=True,
    )

    assert run_data.stats.config_count == 1
    assert run_data.stats.unknown_count == 1
    reason_summary = run_data.stats.reason_legend["state-skip"]
    assert reason_summary.canonical_code == "state_skipped"
    assert reason_summary.message == "reference state skipped: requested by benchmark"
    table = find_tabulate_call(tabulate_calls, INTERVAL_DISPLAY_HEADERS)
    row = table["rows"][0]
    assert row[-4] == "n/a"
    assert row[-3] == "1.000 s"
    assert row[-2] == ""
    assert row[-1] == "\U0001f7e1 ????"


def test_compare_benches_counts_missing_summaries_as_unknown(
    monkeypatch, nvbench_compare
):
    run_data = make_comparison_run_data(nvbench_compare)
    tabulate_calls = capture_tabulate_calls(monkeypatch, nvbench_compare)

    ref_state = make_state(nvbench_compare, "state")
    del ref_state["summaries"]
    cmp_state = make_state(nvbench_compare, "state", mean="1.0")

    nvbench_compare.compare_benches(
        run_data,
        [make_benchmark([ref_state])],
        [make_benchmark([cmp_state])],
        threshold=1.0,
        plot_along=None,
        plot=False,
        dark=False,
        filter_plan=make_filter_plan(nvbench_compare),
        no_color=True,
    )

    assert run_data.stats.config_count == 1
    assert run_data.stats.unknown_count == 1
    reason_summary = run_data.stats.reason_legend["summ-miss"]
    assert reason_summary.canonical_code == "gpu_timing_summaries_missing"
    assert reason_summary.message == "reference GPU timing summaries are missing"
    table = find_tabulate_call(tabulate_calls, INTERVAL_DISPLAY_HEADERS)
    row = table["rows"][0]
    assert row[-4] == "n/a"
    assert row[-3] == "1.000 s"
    assert row[-2] == ""
    assert row[-1] == "\U0001f7e1 ????"


def test_compare_benches_plot_skips_unknown_rows(monkeypatch, nvbench_compare):
    plotted_entries = []

    def fake_plot_comparison_entries(entries, *args, **kwargs):
        plotted_entries.extend(entries)
        return 0

    monkeypatch.setattr(
        nvbench_compare, "plot_comparison_entries", fake_plot_comparison_entries
    )

    run_data = make_comparison_run_data(nvbench_compare)
    ref_state = make_state(nvbench_compare, "state")
    del ref_state["summaries"]
    cmp_state = make_state(nvbench_compare, "state", mean="1.0")

    nvbench_compare.compare_benches(
        run_data,
        [make_benchmark([ref_state])],
        [make_benchmark([cmp_state])],
        threshold=1.0,
        plot_along=None,
        plot=True,
        dark=False,
        filter_plan=make_filter_plan(nvbench_compare),
        no_color=True,
    )

    assert run_data.stats.config_count == 1
    assert run_data.stats.unknown_count == 1
    assert plotted_entries == []


def test_compare_benches_defaults_to_interval_display(monkeypatch, nvbench_compare):
    run_data = make_comparison_run_data(nvbench_compare)
    tabulate_calls = capture_tabulate_calls(monkeypatch, nvbench_compare)

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

    table = find_tabulate_call(tabulate_calls, INTERVAL_DISPLAY_HEADERS)
    row = table["rows"][0]
    assert row[-4].startswith("1.000 s")
    assert row[-3].startswith("1.010 s")
    assert row[-2] == ""


def test_compare_benches_legacy_display_uses_scalar_diff(monkeypatch, nvbench_compare):
    run_data = make_comparison_run_data(nvbench_compare)
    tabulate_calls = capture_tabulate_calls(monkeypatch, nvbench_compare)

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

    table = find_tabulate_call(tabulate_calls, LEGACY_DISPLAY_HEADERS)
    row = table["rows"][0]
    assert row[-7] == "1.000 s"
    assert row[-5] == "1.010 s"
    assert row[-3] == "10.000 ms"
    assert row[-2] == "1.00%"


def test_compare_benches_explain_display_uses_explicit_intervals(
    monkeypatch, nvbench_compare
):
    run_data = make_comparison_run_data(nvbench_compare)
    tabulate_calls = capture_tabulate_calls(monkeypatch, nvbench_compare)

    ref_state = make_state(nvbench_compare, "state", mean="1.0")
    ref_state["summaries"].extend(
        [
            make_summary(nvbench_compare, "GPU_TIME_MIN_TAG", "1.0"),
            make_summary(nvbench_compare, "GPU_TIME_Q1_TAG", "1.01"),
            make_summary(nvbench_compare, "GPU_TIME_MEDIAN_TAG", "1.02"),
            make_summary(nvbench_compare, "GPU_TIME_Q3_TAG", "1.03"),
            make_summary(nvbench_compare, "GPU_TIME_IQR_RELATIVE_TAG", "0.01"),
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
            make_summary(nvbench_compare, "GPU_TIME_IQR_RELATIVE_TAG", "0.01"),
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

    table = find_tabulate_call(tabulate_calls, EXPLAIN_DISPLAY_HEADERS)
    row = table["rows"][0]
    assert row[-7] == "1.0[00 | 20 | 30] s"
    assert row[-6] == "1.0[10 | 30 | 40] s"
    assert row[-3] == "centers-far"
    assert row[-2] == ""


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
