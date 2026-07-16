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
    monkeypatch.delitem(sys.modules, "nvbench_tooling_deps", raising=False)
    return importlib.import_module("nvbench_tooling_deps")


def make_packaged_scripts_tree(tmp_path: Path) -> Path:
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    package_dir = tmp_path / "cuda" / "bench" / "scripts"
    results_dir = tmp_path / "cuda" / "bench" / "results"
    package_dir.mkdir(parents=True)
    results_dir.mkdir()
    for package in [
        tmp_path / "cuda",
        tmp_path / "cuda" / "bench",
        package_dir,
        results_dir,
    ]:
        (package / "__init__.py").write_text("", encoding="utf-8")
    for filename in [
        "nvbench_compare.py",
        "nvbench_compare_robust.py",
        "nvbench_histogram.py",
        "nvbench_json_summary.py",
        "nvbench_plot_bwutil.py",
        "nvbench_tooling_deps.py",
        "nvbench_walltime.py",
    ]:
        shutil.copy(scripts_dir / filename, package_dir / filename)
    shutil.copytree(
        scripts_dir / "nvbench_json",
        package_dir / "nvbench_json",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (results_dir / "__init__.py").write_text(
        "class BenchmarkResult:\n"
        "    pass\n"
        "class BenchmarkResultSummary:\n"
        "    pass\n"
        "class SubBenchmarkResult:\n"
        "    pass\n"
        "class SubBenchmarkState:\n"
        "    pass\n",
        encoding="utf-8",
    )
    return package_dir


def clear_packaged_cuda_modules(monkeypatch):
    for module_name in list(sys.modules):
        if module_name == "cuda" or module_name.startswith("cuda."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)


def test_tooling_deps_imports_from_packaged_script_path(tmp_path, monkeypatch):
    package_dir = make_packaged_scripts_tree(tmp_path)

    monkeypatch.syspath_prepend(str(tmp_path))
    clear_packaged_cuda_modules(monkeypatch)

    module = importlib.import_module("cuda.bench.scripts.nvbench_tooling_deps")

    assert Path(module.__file__) == package_dir / "nvbench_tooling_deps.py"
    assert module.ToolingDependency("math", "math", "testing").extra == "tools"


@pytest.mark.parametrize(
    ("module_name", "expected_entry"),
    [
        ("cuda.bench.scripts.nvbench_compare", "main"),
        ("cuda.bench.scripts.nvbench_compare_robust", "main"),
        ("cuda.bench.scripts.nvbench_histogram", "main"),
        ("cuda.bench.scripts.nvbench_json_summary", "main"),
        ("cuda.bench.scripts.nvbench_plot_bwutil", "main"),
        ("cuda.bench.scripts.nvbench_walltime", "main"),
    ],
)
def test_console_script_modules_import_from_packaged_paths(
    tmp_path, monkeypatch, module_name, expected_entry
):
    package_dir = make_packaged_scripts_tree(tmp_path)
    leaf_module = module_name.rsplit(".", 1)[-1]

    monkeypatch.syspath_prepend(str(tmp_path))
    clear_packaged_cuda_modules(monkeypatch)

    module = importlib.import_module(module_name)

    assert Path(module.__file__) == package_dir / f"{leaf_module}.py"
    assert callable(getattr(module, expected_entry))


def test_compare_console_scripts_are_explicitly_named():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    contents = pyproject.read_text(encoding="utf-8")

    assert (
        'nvbench-compare-robust = "cuda.bench.scripts.nvbench_compare_robust:main"'
        in contents
    )
    assert (
        'nvbench-compare-legacy = "cuda.bench.scripts.nvbench_compare:main"' in contents
    )
    assert 'nvbench-compare = "cuda.bench.scripts.nvbench_compare:main"' not in contents


def test_nvbench_compare_script_path_uses_legacy_behavior(monkeypatch):
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts_dir))
    monkeypatch.delitem(sys.modules, "nvbench_compare", raising=False)

    module = importlib.import_module("nvbench_compare")

    assert [status.value for status in module.ComparisonStatus] == [
        "????",
        "SAME",
        "FAST",
        "SLOW",
    ]
    assert module.get_display_headers()[0] == [
        "Ref Time",
        "Ref Noise",
        "Cmp Time",
        "Cmp Noise",
        "Diff",
        "%Diff",
        "Status",
    ]


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
