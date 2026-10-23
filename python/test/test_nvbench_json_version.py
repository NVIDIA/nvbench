# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def json_version(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("nvbench_json.version")


@pytest.mark.parametrize("version", ["1.0.0", "1.1.0"])
def test_reader_supports_old_and_new_json_versions(json_version, version, capsys):
    json_version.check_file_version(
        "results.json", {"meta": {"version": {"json": {"string": version}}}}
    )
    assert capsys.readouterr().out == ""


def test_supported_script_version(json_version):
    json_version.check_script_version(
        "results.json", {"meta": {"version": {"scripts": {"string": "1.0.0"}}}}
    )


@pytest.mark.parametrize("metadata", [{}, {"meta": {}}, {"meta": {"version": {}}}])
def test_missing_script_version_is_rejected(json_version, metadata):
    with pytest.raises(ValueError, match="missing script compatibility version"):
        json_version.check_script_version("results.json", metadata)


@pytest.mark.parametrize("actual", ["0.9.0", "2.0.0", None, [], {}])
def test_unsupported_script_version_is_rejected(json_version, actual):
    with pytest.raises(ValueError, match="unsupported script compatibility version"):
        json_version.check_script_version(
            "results.json", {"meta": {"version": {"scripts": {"string": actual}}}}
        )
