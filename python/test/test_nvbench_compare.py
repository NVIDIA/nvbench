# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib
import json
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
        lambda path: {
            "meta": {"version": {"scripts": {"string": "1.0.0"}}},
            "devices": [],
            "benchmarks": [],
        },
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


def test_main_rejects_mismatched_timers(nvbench_compare, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(nvbench_compare, "load_nvbench_compare_tooling", lambda: None)
    paths = []
    for timer in ("cuda_events", "cupti"):
        path = tmp_path / f"{timer}.json"
        path.write_text(
            json.dumps(
                {
                    "meta": {"version": {"scripts": {"string": "1.0.0"}}},
                    "devices": [{"id": 0, "name": "GPU"}],
                    "benchmarks": [
                        {
                            "name": "bench",
                            "devices": [0],
                            "axes": [],
                            "states": [
                                {
                                    "name": "state",
                                    "device": 0,
                                    "axis_values": [],
                                    "gpu_timer": timer,
                                    "summaries": [{"tag": "nv/cold/time/gpu/mean"}],
                                }
                            ],
                        }
                    ],
                }
            )
        )
        paths.append(str(path))
    monkeypatch.setattr(sys, "argv", ["nvbench_compare", "--ignore-devices", *paths])
    assert nvbench_compare.main() == 1
    assert "incompatible GPU timing methodologies" in capsys.readouterr().err


@pytest.mark.parametrize(
    "versions", [(None, "1.0.0"), ("1.0.0", "1.1.0"), ("1.0.0", "2.0.0")]
)
def test_main_rejects_incompatible_script_versions(
    nvbench_compare, monkeypatch, capsys, versions
):
    monkeypatch.setattr(nvbench_compare, "load_nvbench_compare_tooling", lambda: None)
    roots = {}
    for filename, version in zip(("ref.json", "cmp.json"), versions):
        root = {"devices": [], "benchmarks": []}
        if version is not None:
            root["meta"] = {"version": {"scripts": {"string": version}}}
        roots[filename] = root
    monkeypatch.setattr(nvbench_compare.reader, "read_file", lambda path: roots[path])
    monkeypatch.setattr(sys, "argv", ["nvbench_compare", "ref.json", "cmp.json"])
    assert nvbench_compare.main() == 1
    assert "script compatibility version" in capsys.readouterr().err
