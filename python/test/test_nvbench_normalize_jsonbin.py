# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
from pathlib import Path

import pytest


@pytest.fixture
def normalizer():
    import importlib.util

    path = Path(__file__).parents[1] / "scripts" / "nvbench_normalize_jsonbin.py"
    spec = importlib.util.spec_from_file_location("nvbench_normalize_jsonbin", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_result(path: Path, filename: str) -> None:
    path.write_text(
        json.dumps(
            {
                "benchmarks": [
                    {
                        "states": [
                            {
                                "summaries": [
                                    {"hint": "file/sample_times", "filename": filename}
                                ]
                            }
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_normalize_legacy_launch_directory_path(tmp_path, normalizer, monkeypatch):
    result_dir = tmp_path / "results"
    launch_dir = tmp_path / "launch"
    result_dir.mkdir()
    launch_dir.mkdir()
    sidecar = launch_dir / "result.json-bin" / "0.bin"
    sidecar.parent.mkdir()
    sidecar.write_bytes(b"data")
    result = result_dir / "result.json"
    make_result(result, "result.json-bin/0.bin")
    monkeypatch.chdir(launch_dir)

    document, changes = normalizer.normalize_jsonbin(result)

    assert changes == [("result.json-bin/0.bin", "../launch/result.json-bin/0.bin")]
    assert (
        document["benchmarks"][0]["states"][0]["summaries"][0]["filename"]
        == changes[0][1]
    )


def test_normalize_rejects_ambiguous_sidecar(tmp_path, normalizer, monkeypatch):
    result = tmp_path / "result.json"
    make_result(result, "result.json-bin/0.bin")
    (tmp_path / "result.json-bin").mkdir()
    (tmp_path / "result.json-bin" / "0.bin").write_bytes(b"json")
    launch_dir = tmp_path / "launch"
    launch_dir.mkdir()
    (launch_dir / "result.json-bin").mkdir()
    (launch_dir / "result.json-bin" / "0.bin").write_bytes(b"launch")
    monkeypatch.chdir(launch_dir)

    with pytest.raises(normalizer.SidecarResolutionError, match="ambiguous"):
        normalizer.normalize_jsonbin(result)
