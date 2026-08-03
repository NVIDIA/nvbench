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

    (tmp_path / "nested.txt").write_text(
        "\n".join(
            [
                "pandas==2.2.0",
                "--index-url https://example.invalid/simple",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "constraints.txt").write_text(
        "\n".join(
            [
                "numpy==2.0.0",
                "-r nested.txt",
            ]
        ),
        encoding="utf-8",
    )

    assert formatter.parse_constraint_names(tmp_path / "constraints.txt") == {
        "numpy": "numpy",
        "pandas": "pandas",
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
