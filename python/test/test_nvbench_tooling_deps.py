# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib
import importlib.metadata
import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

SCRIPT_SOURCE_FILES = [
    "_nvbench_compare_plotting.py",
    "nvbench_compare.py",
    "nvbench_compare_robust.py",
    "nvbench_histogram.py",
    "nvbench_json_summary.py",
    "nvbench_plot_bwutil.py",
    "nvbench_tooling_deps.py",
    "nvbench_walltime.py",
]


@pytest.fixture
def tooling_deps(monkeypatch):
    scripts_dir = find_script_sources()
    monkeypatch.syspath_prepend(str(scripts_dir))
    monkeypatch.delitem(sys.modules, "nvbench_tooling_deps", raising=False)
    return importlib.import_module("nvbench_tooling_deps")


def is_script_source_dir(path: Path) -> bool:
    return (
        all((path / filename).is_file() for filename in SCRIPT_SOURCE_FILES)
        and (path / "nvbench_json").is_dir()
    )


def find_script_sources() -> Path:
    source_tree_scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if source_tree_scripts_dir.is_dir() and is_script_source_dir(
        source_tree_scripts_dir
    ):
        return source_tree_scripts_dir

    try:
        packaged_spec = importlib.util.find_spec(
            "cuda.bench.scripts.nvbench_tooling_deps"
        )
    except ModuleNotFoundError:
        packaged_spec = None
    if packaged_spec is not None and packaged_spec.origin is not None:
        scripts_dir = Path(packaged_spec.origin).resolve().parent
        if is_script_source_dir(scripts_dir):
            return scripts_dir

    pytest.skip("NVBench script sources are not available in this test layout")


def make_packaged_scripts_tree(tmp_path: Path, scripts_dir: Path) -> Path:
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
    for filename in SCRIPT_SOURCE_FILES:
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


def assert_packaged_script_module(module, module_name: str, filename: str) -> None:
    # Editable installs may redirect cuda.bench.scripts.* to the source-tree
    # scripts/ directory, so assert the import contract rather than a physical
    # package directory.
    assert module.__name__ == module_name
    assert module.__package__ == "cuda.bench.scripts"
    assert Path(module.__file__).name == filename


def test_tooling_deps_imports_from_packaged_script_path(tmp_path, monkeypatch):
    scripts_dir = find_script_sources()
    make_packaged_scripts_tree(tmp_path, scripts_dir)

    monkeypatch.syspath_prepend(str(tmp_path))
    clear_packaged_cuda_modules(monkeypatch)

    module_name = "cuda.bench.scripts.nvbench_tooling_deps"
    module = importlib.import_module(module_name)

    assert_packaged_script_module(module, module_name, "nvbench_tooling_deps.py")
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
    scripts_dir = find_script_sources()
    make_packaged_scripts_tree(tmp_path, scripts_dir)
    leaf_module = module_name.rsplit(".", 1)[-1]

    monkeypatch.syspath_prepend(str(tmp_path))
    clear_packaged_cuda_modules(monkeypatch)

    module = importlib.import_module(module_name)

    assert_packaged_script_module(module, module_name, f"{leaf_module}.py")
    assert callable(getattr(module, expected_entry))


def test_compare_console_scripts_are_explicitly_named():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if not pyproject.exists():
        entry_points = importlib.metadata.entry_points(group="console_scripts")
        scripts = {entry_point.name: entry_point.value for entry_point in entry_points}

        assert (
            scripts["nvbench-compare-robust"]
            == "cuda.bench.scripts.nvbench_compare_robust:main"
        )
        assert (
            scripts["nvbench-compare-legacy"]
            == "cuda.bench.scripts.nvbench_compare:main"
        )
        assert "nvbench-compare" not in scripts
        return

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
    scripts_dir = find_script_sources()
    monkeypatch.syspath_prepend(str(scripts_dir))
    monkeypatch.delitem(sys.modules, "nvbench_compare", raising=False)

    module = importlib.import_module("nvbench_compare")

    assert hasattr(module, "compare_benches")


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
