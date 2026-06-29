# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib
import shutil
import sys
from pathlib import Path

import pytest


@pytest.fixture
def tooling_deps(monkeypatch):
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts_dir))
    sys.modules.pop("nvbench_tooling_deps", None)
    return importlib.import_module("nvbench_tooling_deps")


def test_tooling_deps_imports_from_packaged_script_path(tmp_path, monkeypatch):
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    package_dir = tmp_path / "cuda" / "bench" / "scripts"
    package_dir.mkdir(parents=True)
    for package in [tmp_path / "cuda", tmp_path / "cuda" / "bench", package_dir]:
        (package / "__init__.py").write_text("", encoding="utf-8")
    shutil.copy(
        scripts_dir / "nvbench_tooling_deps.py",
        package_dir / "nvbench_tooling_deps.py",
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("cuda.bench.scripts.nvbench_tooling_deps", None)

    module = importlib.import_module("cuda.bench.scripts.nvbench_tooling_deps")

    assert module.ToolingDependency("math", "math", "testing").extra == "tools"


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


def test_require_tooling_dependency_reraises_broken_package_import_error(
    tmp_path, monkeypatch, tooling_deps
):
    package_dir = tmp_path / "nvbench_broken_tooling_dependency_for_test"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text(
        "raise ImportError('broken package')\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ImportError, match="broken package"):
        tooling_deps.require_tooling_dependency(
            tooling_deps.ToolingDependency(
                package_dir.name, package_dir.name, "testing broken imports"
            ),
            tool_name="test-tool",
        )


def test_require_tooling_dependency_reraises_transitive_module_not_found(
    tmp_path, monkeypatch, tooling_deps
):
    package_dir = tmp_path / "nvbench_transitive_missing_dependency_for_test"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text(
        "import nvbench_missing_transitive_dependency_for_test\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ModuleNotFoundError) as exc_info:
        tooling_deps.require_tooling_dependency(
            tooling_deps.ToolingDependency(
                package_dir.name, package_dir.name, "testing transitive imports"
            ),
            tool_name="test-tool",
        )

    assert exc_info.value.name == "nvbench_missing_transitive_dependency_for_test"
