# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import os
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


def test_sidecar_root_selects_matching_file_from_ambiguous_candidates(
    tmp_path, normalizer, monkeypatch
):
    result = tmp_path / "result.json"
    make_result(result, "result.json-bin/0.bin")
    local = tmp_path / "result.json-bin" / "0.bin"
    local.parent.mkdir()
    local.write_bytes(b"local")
    launch_dir = tmp_path / "launch"
    launch_dir.mkdir()
    selected = launch_dir / "result.json-bin" / "0.bin"
    selected.parent.mkdir()
    selected.write_bytes(b"selected")
    monkeypatch.chdir(launch_dir)

    resolved = normalizer.resolve_sidecar(
        "result.json-bin/0.bin", result, sidecar_root=launch_dir
    )

    assert resolved == selected.resolve()


def test_unrelated_prefix_does_not_fall_back_to_json_sidecar(tmp_path, normalizer):
    result = tmp_path / "result.json"
    make_result(result, "other/result.json-bin/0.bin")
    local = tmp_path / "result.json-bin" / "0.bin"
    local.parent.mkdir()
    local.write_bytes(b"unrelated")

    with pytest.raises(normalizer.SidecarResolutionError, match="could not resolve"):
        normalizer.normalize_jsonbin(result)


def test_unhashable_unrelated_hint_is_ignored(tmp_path, normalizer):
    result = tmp_path / "result.json"
    result.write_text(json.dumps({"metadata": {"hint": [], "value": 1}}))

    document, changes = normalizer.normalize_jsonbin(result)

    assert document["metadata"]["hint"] == []
    assert changes == []


def test_output_rebases_sidecar_path_from_output_directory(tmp_path, normalizer):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output" / "nested"
    sidecar_dir = tmp_path / "sidecars"
    input_dir.mkdir()
    output_dir.mkdir(parents=True)
    sidecar = sidecar_dir / "0.bin"
    sidecar_dir.mkdir()
    sidecar.write_bytes(b"data")
    result = input_dir / "result.json"
    make_result(result, "../sidecars/0.bin")
    output = output_dir / "normalized.json"

    document, changes = normalizer.normalize_jsonbin(result, output_path=output)

    expected = os.path.relpath(sidecar.resolve(), output_dir.resolve()).replace(
        os.sep, "/"
    )
    assert (
        document["benchmarks"][0]["states"][0]["summaries"][0]["filename"] == expected
    )
    assert changes == [("../sidecars/0.bin", expected)]


def test_output_copies_unchanged_json(tmp_path, normalizer):
    sidecar = tmp_path / "result.json-bin" / "0.bin"
    sidecar.parent.mkdir()
    sidecar.write_bytes(b"data")
    result = tmp_path / "result.json"
    make_result(result, "result.json-bin/0.bin")
    output = tmp_path / "copy.json"

    assert normalizer.main([str(result), "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == json.loads(
        result.read_text(encoding="utf-8")
    )


def test_output_rejects_input_path_without_overwriting(tmp_path, normalizer):
    sidecar = tmp_path / "result.json-bin" / "0.bin"
    sidecar.parent.mkdir()
    sidecar.write_bytes(b"data")
    result = tmp_path / "result.json"
    make_result(result, "result.json-bin/0.bin")
    original = result.read_bytes()

    assert normalizer.main([str(result), "--output", str(result)]) == 2
    assert result.read_bytes() == original
