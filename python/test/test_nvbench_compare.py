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
