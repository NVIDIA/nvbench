# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib
import sys
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


def make_root(time, noise):
    def summary(tag, value):
        return {
            "tag": tag,
            "data": [{"name": "value", "type": "float64", "value": value}],
        }

    return {
        "devices": [{"id": 0, "name": "GPU"}],
        "benchmarks": [
            {
                "name": "bench",
                "devices": [0],
                "axes": [],
                "states": [
                    {
                        "name": "Device=0",
                        "device": 0,
                        "axis_values": [],
                        "summaries": [
                            summary("nv/cold/time/gpu/mean", time),
                            summary("nv/cold/time/gpu/stdev/relative", noise),
                        ],
                    }
                ],
            }
        ],
    }


@pytest.mark.parametrize(
    ("ref_noise", "cmp_noise", "expected"),
    [
        ("inf", "inf", (0, 1, 0)),
        (None, None, (0, 1, 0)),
        ("inf", "0.01", (0, 0, 1)),
        ("0.01", "inf", (0, 0, 1)),
        ("0.01", "0.01", (0, 0, 1)),
    ],
)
def test_compare_infinite_noise_is_unknown(
    nvbench_compare, monkeypatch, capsys, ref_noise, cmp_noise, expected
):
    class StubTabulate:
        @staticmethod
        def tabulate(rows, **kwargs):
            return "\n".join(" | ".join(map(str, row)) for row in rows)

    monkeypatch.setattr(nvbench_compare, "load_nvbench_compare_tooling", lambda: None)
    monkeypatch.setattr(nvbench_compare, "load_tabulate_for_table_output", lambda: None)
    monkeypatch.setattr(nvbench_compare, "tabulate", StubTabulate)
    monkeypatch.setattr(nvbench_compare, "tabulate_version", (0, 9, 0))
    roots = {
        "reference.json": make_root("0.001", ref_noise),
        "compare.json": make_root("0.002", cmp_noise),
    }
    monkeypatch.setattr(nvbench_compare.reader, "read_file", roots.__getitem__)
    for name in ("config_count", "pass_count", "unknown_count", "failure_count"):
        monkeypatch.setattr(nvbench_compare, name, 0)
    monkeypatch.setattr(
        sys, "argv", ["nvbench_compare", "reference.json", "compare.json"]
    )

    nvbench_compare.main()

    assert (
        nvbench_compare.pass_count,
        nvbench_compare.unknown_count,
        nvbench_compare.failure_count,
    ) == expected
    assert "inf%" not in capsys.readouterr().out
