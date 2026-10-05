# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

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

    assert tooling_calls == [{"use_color": False}]
    assert len(compare_calls[0]) == 9
    assert (
        nvbench_compare.format_status("SAME", nvbench_compare.Emoji.BLUE) == "🔵 SAME"
    )


def test_color_flag_is_forwarded_to_output_and_dependency_loader(
    nvbench_compare, monkeypatch
):
    tooling_calls, compare_calls = run_main(
        nvbench_compare, monkeypatch, "--color", "reference.json", "compare.json"
    )

    assert tooling_calls == [{"use_color": True}]
    assert compare_calls[0][-1] is True


def test_colorama_is_loaded_only_when_color_is_requested(nvbench_compare, monkeypatch):
    loaded = []

    def require_tooling_dependency(dependency, *, tool_name):
        loaded.append(dependency.import_name)
        if dependency.import_name == "colorama":
            return SimpleNamespace(
                Fore=SimpleNamespace(
                    YELLOW="yellow",
                    BLUE="blue",
                    GREEN="green",
                    RED="red",
                    RESET="reset",
                )
            )
        return SimpleNamespace(__version__="1.0.0")

    monkeypatch.setattr(
        nvbench_compare, "require_tooling_dependency", require_tooling_dependency
    )
    monkeypatch.setattr(nvbench_compare, "tabulate", None)
    monkeypatch.setattr(nvbench_compare, "Fore", None)

    nvbench_compare.load_nvbench_compare_tooling(use_color=False)
    assert "colorama" not in loaded

    monkeypatch.setattr(nvbench_compare, "tabulate", None)
    nvbench_compare.load_nvbench_compare_tooling(use_color=True)
    assert loaded.count("colorama") == 1
    assert (
        nvbench_compare.format_status(
            "SAME", nvbench_compare.Emoji.BLUE, use_color=True
        )
        == "blueSAMEreset"
    )
