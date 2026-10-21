# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib
import json
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def nvbench_walltime(monkeypatch):
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts_dir))
    monkeypatch.delitem(sys.modules, "nvbench_walltime", raising=False)
    return importlib.import_module("nvbench_walltime")


def walltime_summary(name, value):
    return {
        "tag": "nv/{}/walltime".format(name),
        "name": "Walltime",
        "hint": "duration",
        "data": [{"name": "value", "type": "float64", "value": value}],
    }


def write_result_json(path):
    path.write_text(
        json.dumps(
            {
                "meta": {"version": {"json": {"string": "1.0.0"}}},
                "devices": [{"id": 0, "name": "Test GPU"}],
                "benchmarks": [
                    {
                        "name": "copy",
                        "devices": [0],
                        "axes": [
                            {
                                "name": "T",
                                "type": "type",
                                "flags": "",
                                "values": [
                                    {"input_string": "I8", "description": "int8_t"},
                                    {"input_string": "I16", "description": "int16_t"},
                                ],
                            }
                        ],
                        "states": [
                            {
                                "name": "Device=0 T=I8",
                                "device": 0,
                                "axis_values": [
                                    {"name": "T", "type": "type", "value": "I8"}
                                ],
                                "summaries": None,
                                "is_skipped": True,
                                "skip_reason": "Unsupported type",
                            },
                            {
                                "name": "Device=0 T=I16",
                                "device": 0,
                                "axis_values": [
                                    {"name": "T", "type": "type", "value": "I16"}
                                ],
                                "summaries": [
                                    walltime_summary("cold", "1.5"),
                                    walltime_summary("batch", "0.5"),
                                ],
                                "is_skipped": False,
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_walltime_reports_skipped_states_as_zero(
    nvbench_walltime, monkeypatch, tmp_path
):
    json_path = tmp_path / "result.json"
    write_result_json(json_path)

    tables = []
    fake_tabulate = types.SimpleNamespace(
        __version__="0.9.0",
        tabulate=lambda rows, headers, **kwargs: tables.append((headers, rows)) or "",
    )
    monkeypatch.setattr(nvbench_walltime, "tabulate", fake_tabulate)
    monkeypatch.setattr(nvbench_walltime, "tabulate_version", (0, 9, 0))
    monkeypatch.setattr(sys, "argv", ["nvbench_walltime", str(json_path)])

    assert nvbench_walltime.main() == 0

    state_table = next(
        rows for headers, rows in tables if headers[0] == "Configuration"
    )
    assert state_table == [
        ["Device=0 T=I8", "0.00%", "0.000", "0.000", "0.000"],
        ["Device=0 T=I16", "100.00%", "2.000", "1.500", "0.500"],
        ["Total", "100.00%", "2.000", "1.500", "0.500"],
    ]
