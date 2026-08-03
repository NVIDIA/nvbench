from __future__ import annotations

import importlib.util
from pathlib import Path


def load_formatter_module():
    module_path = Path(__file__).with_name("format_python_example_constraints.py")
    spec = importlib.util.spec_from_file_location(
        "format_python_example_constraints", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_constraint_names_follows_requirement_includes(tmp_path):
    formatter = load_formatter_module()

    (tmp_path / "nested-a.txt").write_text(
        "\n".join(
            [
                "pandas==2.2.0",
                "--index-url https://example.invalid/simple",
                "--requirement=nested-b.txt",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "nested-b.txt").write_text("tabulate==0.9.0\n", encoding="utf-8")
    (tmp_path / "nested-c.txt").write_text("ruff==0.9.0\n", encoding="utf-8")
    (tmp_path / "constraints.txt").write_text(
        "\n".join(
            [
                "numpy==2.0.0",
                "-r nested-a.txt",
                "-rnested-c.txt",
            ]
        ),
        encoding="utf-8",
    )

    assert formatter.parse_constraint_names(tmp_path / "constraints.txt") == {
        "numpy": "numpy",
        "pandas": "pandas",
        "ruff": "ruff",
        "tabulate": "tabulate",
    }


def test_parse_constraint_names_ignores_recursive_include_cycle(tmp_path):
    formatter = load_formatter_module()

    (tmp_path / "a.txt").write_text(
        "\n".join(
            [
                "numpy==2.0.0",
                "-r b.txt",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "b.txt").write_text(
        "\n".join(
            [
                "pandas==2.2.0",
                "-r a.txt",
            ]
        ),
        encoding="utf-8",
    )

    assert formatter.parse_constraint_names(tmp_path / "a.txt") == {
        "numpy": "numpy",
        "pandas": "pandas",
    }


def test_constraints_candidate_reports_direct_references(tmp_path, capsys):
    formatter = load_formatter_module()

    constraints_file = tmp_path / "constraints.txt"
    freeze_dir = tmp_path / "freeze"
    freeze_dir.mkdir()

    constraints_file.write_text(
        "\n".join(
            [
                "cuda-bench==0.1.0",
                "numpy==2.0.0",
            ]
        ),
        encoding="utf-8",
    )
    (freeze_dir / "core.txt").write_text(
        "\n".join(
            [
                "cuda-bench @ file:///tmp/cuda_bench-0.1.0.whl",
                "numpy==2.0.0",
            ]
        ),
        encoding="utf-8",
    )

    formatter.print_constraints_candidate(
        constraints_file=constraints_file,
        freeze_dir=freeze_dir,
        python_version="3.13",
        cuda_extra="cu13",
    )

    output = capsys.readouterr().out
    assert (
        "# DIRECT cuda-bench: cuda-bench @ file:///tmp/cuda_bench-0.1.0.whl" in output
    )
    assert "# MISSING cuda-bench" not in output
