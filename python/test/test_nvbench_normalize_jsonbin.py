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
                                    {
                                        "hint": "file/sample_times",
                                        "data": [
                                            {
                                                "name": "filename",
                                                "type": "string",
                                                "value": filename,
                                            },
                                            {
                                                "name": "size",
                                                "type": "int64",
                                                "value": "4",
                                            },
                                        ],
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def summary_filename(document: dict) -> str:
    return document["benchmarks"][0]["states"][0]["summaries"][0]["data"][0]["value"]


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
    assert summary_filename(document) == changes[0][1]


def test_normalize_legacy_inline_filename_record(tmp_path, normalizer):
    sidecar = tmp_path / "sidecars" / "0.bin"
    sidecar.parent.mkdir()
    sidecar.write_bytes(b"data")
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {"summary": {"hint": "file/sample_times", "filename": str(sidecar)}}
        ),
        encoding="utf-8",
    )

    document, changes = normalizer.normalize_jsonbin(result)

    assert document["summary"]["filename"] == "sidecars/0.bin"
    assert changes == [(str(sidecar), "sidecars/0.bin")]


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


def test_resolve_sidecar_preserves_symlink_before_parent_traversal(
    tmp_path, normalizer, monkeypatch
):
    launch_dir = tmp_path / "launch"
    launch_dir.mkdir()
    result = launch_dir / "result.json"
    make_result(result, "link/../foo/result.json-freqs-bin/0.bin")

    target_dir = tmp_path / "other" / "nested"
    target_dir.mkdir(parents=True)
    (target_dir.parent / "foo" / "result.json-freqs-bin").mkdir(parents=True)
    selected = target_dir.parent / "foo" / "result.json-freqs-bin" / "0.bin"
    selected.write_bytes(b"selected")
    (launch_dir / "link").symlink_to(target_dir, target_is_directory=True)
    monkeypatch.chdir(launch_dir)

    filename = "link/../foo/result.json-freqs-bin/0.bin"
    assert normalizer.resolve_sidecar(filename, result) == selected.resolve()
    assert (
        normalizer.resolve_sidecar(filename, result, sidecar_root=launch_dir)
        == selected.resolve()
    )


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
    assert summary_filename(document) == expected
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


def test_dry_run_reports_no_changes_for_normalized_json(tmp_path, normalizer, capsys):
    sidecar = tmp_path / "result.json-bin" / "0.bin"
    sidecar.parent.mkdir()
    sidecar.write_bytes(b"data")
    result = tmp_path / "result.json"
    make_result(result, "result.json-bin/0.bin")

    assert normalizer.main([str(result), "--dry-run"]) == 0
    assert capsys.readouterr().out.strip() == f"No changes needed: {result}"


def test_output_rejects_input_path_without_overwriting(tmp_path, normalizer):
    sidecar = tmp_path / "result.json-bin" / "0.bin"
    sidecar.parent.mkdir()
    sidecar.write_bytes(b"data")
    result = tmp_path / "result.json"
    make_result(result, "result.json-bin/0.bin")
    original = result.read_bytes()

    assert normalizer.main([str(result), "--output", str(result)]) == 2
    assert result.read_bytes() == original


def test_output_rejects_hard_link_to_input_without_overwriting(
    tmp_path, normalizer, monkeypatch
):
    sidecar_dir = tmp_path / "sidecars"
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    sidecar_dir.mkdir()
    input_dir.mkdir()
    output_dir.mkdir()
    (sidecar_dir / "0.bin").write_bytes(b"data")
    result = input_dir / "result.json"
    make_result(result, "sidecars/0.bin")
    output = output_dir / "linked.json"
    try:
        os.link(result, output)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"hard links are unavailable: {exc}")
    original = result.read_bytes()
    monkeypatch.chdir(tmp_path)

    assert normalizer.main([str(result), "--output", str(output)]) == 2
    assert result.read_bytes() == original
    assert output.read_bytes() == original


def test_output_preserves_existing_file_when_atomic_replace_fails(
    tmp_path, normalizer, monkeypatch
):
    sidecar = tmp_path / "sidecars" / "0.bin"
    sidecar.parent.mkdir()
    sidecar.write_bytes(b"data")
    result_dir = tmp_path / "input"
    result_dir.mkdir()
    result = result_dir / "result.json"
    make_result(result, "../sidecars/0.bin")
    output = tmp_path / "output.json"
    output.write_text("existing output\n", encoding="utf-8")
    original_output = output.read_bytes()
    temporary_files = list(tmp_path.glob(f".{output.name}.*"))
    assert temporary_files == []

    def fail_replace(*args):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(normalizer.os, "replace", fail_replace)

    assert normalizer.main([str(result), "--output", str(output)]) == 2
    assert output.read_bytes() == original_output
    assert list(tmp_path.glob(f".{output.name}.*")) == []


def test_in_place_refuses_to_overwrite_existing_backup(
    tmp_path, normalizer, monkeypatch
):
    sidecar = tmp_path / "sidecars" / "0.bin"
    sidecar.parent.mkdir()
    sidecar.write_bytes(b"data")
    result_dir = tmp_path / "input"
    result_dir.mkdir()
    result = result_dir / "result.json"
    make_result(result, str(sidecar.resolve()))
    backup = result.with_name(result.name + ".bak")
    backup.write_bytes(b"preserve this backup")
    original = result.read_bytes()
    original_backup = backup.read_bytes()
    monkeypatch.chdir(tmp_path)

    assert normalizer.main([str(result), "--in-place"]) == 2
    assert result.read_bytes() == original
    assert backup.read_bytes() == original_backup


def test_in_place_normalizes_after_creating_backup(tmp_path, normalizer, monkeypatch):
    sidecar = tmp_path / "sidecars" / "0.bin"
    sidecar.parent.mkdir()
    sidecar.write_bytes(b"data")
    result_dir = tmp_path / "input"
    result_dir.mkdir()
    result = result_dir / "result.json"
    make_result(result, str(sidecar.resolve()))
    original = result.read_bytes()
    monkeypatch.chdir(tmp_path)

    assert normalizer.main([str(result), "--in-place"]) == 0
    assert summary_filename(json.loads(result.read_text(encoding="utf-8"))) == (
        "../sidecars/0.bin"
    )
    assert result.with_name(result.name + ".bak").read_bytes() == original
