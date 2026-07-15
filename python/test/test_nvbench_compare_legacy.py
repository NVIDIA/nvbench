# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib.util
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def nvbench_compare_legacy(monkeypatch):
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
