# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def nvbench_compare(monkeypatch):
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts_dir))
    monkeypatch.delitem(sys.modules, "nvbench_compare", raising=False)
    return importlib.import_module("nvbench_compare")


def run_main(nvbench_compare, monkeypatch, *arguments):
    tooling_calls = []
    compare_calls = []

    monkeypatch.setattr(
        nvbench_compare,
        "load_nvbench_compare_tooling",
        lambda **kwargs: tooling_calls.append(kwargs),
    )
    monkeypatch.setattr(
        nvbench_compare.reader,
        "read_file",
        lambda path: {"devices": [], "benchmarks": []},
    )
    monkeypatch.setattr(
        nvbench_compare,
        "compare_benches",
        lambda *args: compare_calls.append(args),
    )
    monkeypatch.setattr(sys, "argv", ["nvbench_compare", *arguments])

    assert nvbench_compare.main() == 0
    return tooling_calls, compare_calls


def test_compare_uses_emoji_statuses(nvbench_compare, monkeypatch):
    tooling_calls, compare_calls = run_main(
        nvbench_compare, monkeypatch, "reference.json", "compare.json"
    )

    assert tooling_calls == [{}]
    assert len(compare_calls[0]) == 8
    assert (
        nvbench_compare.format_status("SAME", nvbench_compare.Emoji.BLUE) == "🔵 SAME"
    )


def make_state(device, elements, mean):
    def summary(tag, value):
        return {
            "tag": tag,
            "data": [{"name": "value", "type": "float64", "value": str(value)}],
        }

    return {
        "name": f"Device={device} Elements={elements}",
        "device": device,
        "axis_values": [{"name": "Elements", "type": "int64", "value": str(elements)}],
        "summaries": [
            summary("nv/cold/time/gpu/mean", mean),
            summary("nv/cold/time/gpu/stdev/relative", 0.01),
        ],
    }


def make_benchmark(devices, states):
    return {
        "name": "copy",
        "devices": devices,
        "axes": [{"name": "Elements", "type": "int64", "flags": ""}],
        "states": states,
    }


def compare_with_fake_tabulate(nvbench_compare, monkeypatch, devices, ref, cmp):
    tables = []

    def fake_tabulate(rows, headers, **kwargs):
        tables.append(rows)
        return ""

    monkeypatch.setattr(
        nvbench_compare, "tabulate", types.SimpleNamespace(tabulate=fake_tabulate)
    )
    monkeypatch.setattr(nvbench_compare, "tabulate_version", (0, 8, 10))
    monkeypatch.setattr(nvbench_compare, "all_ref_devices", devices)
    monkeypatch.setattr(nvbench_compare, "all_cmp_devices", devices)

    nvbench_compare.compare_benches([ref], [cmp], 0.0, None, False, False, [], [])
    return tables


def test_compare_benches_reports_each_state_under_its_own_device(
    nvbench_compare, monkeypatch, capsys
):
    devices = [{"id": 0, "name": "GPU A"}, {"id": 1, "name": "GPU B"}]
    ref = make_benchmark([0, 1], [make_state(0, 16, 1.0), make_state(1, 16, 2.0)])
    cmp = make_benchmark([0, 1], [make_state(0, 16, 1.5), make_state(1, 16, 3.0)])

    tables = compare_with_fake_tabulate(nvbench_compare, monkeypatch, devices, ref, cmp)

    output = capsys.readouterr().out
    assert "## [0] GPU A\n" in output
    assert "## [1] GPU B\n" in output
    assert " vs. " not in output
    assert [[row[1] for row in rows] for rows in tables] == [["1.000 s"], ["2.000 s"]]
    assert nvbench_compare.config_count == 2


def test_compare_benches_skips_trailing_state_missing_from_reference(
    nvbench_compare, monkeypatch, capsys
):
    devices = [{"id": 0, "name": "GPU A"}]
    ref = make_benchmark([0], [make_state(0, 16, 1.0)])
    cmp = make_benchmark([0], [make_state(0, 16, 1.5), make_state(0, 32, 3.0)])

    tables = compare_with_fake_tabulate(nvbench_compare, monkeypatch, devices, ref, cmp)

    assert "## [0] GPU A\n" in capsys.readouterr().out
    assert [len(rows) for rows in tables] == [1]
    assert nvbench_compare.config_count == 1
