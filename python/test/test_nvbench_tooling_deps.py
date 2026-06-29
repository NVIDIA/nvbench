# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib
import sys
from pathlib import Path

import pytest


@pytest.fixture
def tooling_deps(monkeypatch):
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts_dir))
    sys.modules.pop("nvbench_tooling_deps", None)
    return importlib.import_module("nvbench_tooling_deps")


def test_require_tooling_dependency_returns_loaded_module(tooling_deps):
    module = tooling_deps.require_tooling_dependency(
        tooling_deps.ToolingDependency("math", "math", "testing"),
        tool_name="test-tool",
    )

    assert module.sqrt(9.0) == pytest.approx(3.0)


def test_require_tooling_dependency_reports_install_recipe(tooling_deps):
    missing_name = "nvbench_missing_tooling_dependency_for_test"
    with pytest.raises(tooling_deps.MissingToolingDependencyError) as exc_info:
        tooling_deps.require_tooling_dependency(
            tooling_deps.ToolingDependency(
                missing_name, missing_name, "testing graceful failures"
            ),
            tool_name="test-tool",
        )

    message = str(exc_info.value)
    assert "test-tool requires" in message
    assert "testing graceful failures" in message
    assert "python -m pip install 'cuda-bench[tools]'" in message
    assert missing_name in message
