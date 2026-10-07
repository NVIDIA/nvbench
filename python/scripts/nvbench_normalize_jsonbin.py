# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Normalize legacy NVBench jsonbin sidecar paths."""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

SIDECAR_HINTS = {"file/sample_times", "file/sample_freqs"}


class SidecarResolutionError(ValueError):
    """Raised when a sidecar cannot be resolved unambiguously."""


def _candidate_paths(filename: str, json_path: Path) -> list[Path]:
    path = Path(filename)
    if path.is_absolute():
        candidates = [path]
    else:
        candidates = [json_path.parent / path, Path.cwd() / path]
        # Older NVBench files sometimes recorded a path that included the
        # jsonbin directory while resolving it from the launch directory.
        parts = path.parts
        json_sidecar_names = {json_path.name + "-bin", json_path.name + "-freqs-bin"}
        if parts and parts[0] in json_sidecar_names:
            candidates.append(json_path.parent.joinpath(*parts))

    return candidates


def resolve_sidecar(
    filename: str, json_path: Path, sidecar_root: Path | None = None
) -> Path:
    """Resolve one sidecar filename, rejecting missing or ambiguous matches."""
    path = Path(filename)
    if sidecar_root is not None and not path.is_absolute():
        explicit = sidecar_root / path
        if explicit.is_file():
            return Path(os.path.realpath(os.path.abspath(explicit)))

    matches: list[tuple[Path, Path]] = []
    for candidate in _candidate_paths(filename, json_path):
        if candidate.is_file():
            absolute = Path(os.path.abspath(candidate))
            resolved = Path(os.path.realpath(absolute))
            if all(resolved != existing_resolved for _, existing_resolved in matches):
                matches.append((absolute, resolved))

    if not matches:
        raise SidecarResolutionError(
            f"could not resolve sidecar {filename!r} referenced by {json_path}"
        )
    if len(matches) > 1:
        choices = ", ".join(str(path) for path, _ in matches)
        raise SidecarResolutionError(
            f"ambiguous sidecar {filename!r} referenced by {json_path}: {choices}"
        )
    return matches[0][1]


def _iter_sidecar_records(value: Any):
    if isinstance(value, dict):
        hint = value.get("hint")
        if (
            isinstance(hint, str)
            and hint in SIDECAR_HINTS
            and isinstance(value.get("filename"), str)
        ):
            yield value
        for child in value.values():
            yield from _iter_sidecar_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_sidecar_records(child)


def normalize_jsonbin(
    json_path: Path,
    *,
    sidecar_root: Path | None = None,
    output_path: Path | None = None,
) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    """Return normalized JSON and filename changes for one result file."""
    document = json.loads(json_path.read_text(encoding="utf-8"))
    changes = []
    for record in _iter_sidecar_records(document):
        old_name = record["filename"]
        resolved = resolve_sidecar(old_name, json_path, sidecar_root)
        output_dir = output_path.parent if output_path is not None else json_path.parent
        output_dir = os.path.realpath(os.path.abspath(output_dir))
        new_name = os.path.relpath(resolved, output_dir).replace(os.sep, "/")
        if new_name != old_name:
            record["filename"] = new_name
            changes.append((old_name, new_name))
    return document, changes


def _write_result(json_path: Path, document: dict[str, Any], output: Path) -> None:
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run the jsonbin path normalizer CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_file", type=Path)
    parser.add_argument("--sidecar-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--in-place", action="store_true")
    destination.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        if args.output is not None and os.path.realpath(
            os.path.abspath(args.output)
        ) == os.path.realpath(os.path.abspath(args.json_file)):
            raise ValueError("--output must not name the input JSON file")

        document, changes = normalize_jsonbin(
            args.json_file, sidecar_root=args.sidecar_root, output_path=args.output
        )
        if args.dry_run:
            for old_name, new_name in changes:
                print(f"{old_name} -> {new_name}")
            return 0
        if args.output is not None:
            _write_result(args.json_file, document, args.output)
        elif not changes:
            print(f"No changes needed: {args.json_file}")
            return 0
        elif args.in_place:
            backup = args.json_file.with_name(args.json_file.name + ".bak")
            shutil.copy2(args.json_file, backup)
            _write_result(args.json_file, document, args.json_file)
        else:
            parser.error(
                "choose --dry-run, --in-place, or --output when changes are needed"
            )
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"nvbench-normalize-jsonbin: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
