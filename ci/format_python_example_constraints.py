#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import argparse
import re
from pathlib import Path

PIN_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)\s*==\s*(\S+)\s*$")
REQ_NAME_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)")


def normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_constraint_names(path: Path) -> dict[str, str]:
    names: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped or stripped.startswith("-"):
            continue
        match = REQ_NAME_RE.match(stripped)
        if match:
            original_name = match.group(1)
            names[normalize_name(original_name)] = original_name
    return names


def parse_freeze_file(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = PIN_RE.match(line)
        if match:
            pins[normalize_name(match.group(1))] = f"{match.group(1)}=={match.group(2)}"
    return pins


def parse_requirement_names(path: Path, seen: set[Path] | None = None) -> set[str]:
    if seen is None:
        seen = set()
    path = path.resolve()
    if path in seen:
        return set()
    seen.add(path)

    names: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.split("#", 1)[0].strip()
        if not stripped:
            continue
        if stripped.startswith(("-r ", "--requirement ")):
            _, include_path = stripped.split(maxsplit=1)
            names.update(parse_requirement_names(path.parent / include_path, seen))
            continue
        if stripped.startswith("-"):
            continue

        req = stripped.split(";", 1)[0].strip()
        req = req.split("[", 1)[0].strip()
        match = REQ_NAME_RE.match(req)
        if match:
            names.add(normalize_name(match.group(1)))
    return names


def collect_freeze_pins(freeze_dir: Path) -> dict[str, dict[str, set[str]]]:
    package_to_envs: dict[str, dict[str, set[str]]] = {}
    for freeze_file in sorted(freeze_dir.glob("*.txt")):
        env_name = freeze_file.stem
        for package_name, pin in parse_freeze_file(freeze_file).items():
            envs = package_to_envs.setdefault(package_name, {})
            envs.setdefault(pin, set()).add(env_name)
    return package_to_envs


def print_constraints_candidate(
    *,
    constraints_file: Path,
    freeze_dir: Path,
    python_version: str,
    cuda_extra: str,
) -> None:
    tracked_names = parse_constraint_names(constraints_file)
    package_to_envs = collect_freeze_pins(freeze_dir)

    print("::group::Constraints candidate for tracked Python example packages")
    print(f"# Source: {constraints_file}")
    print("# Generated from floating example dependency run")
    print(f"# Python: {python_version}")
    print(f"# CUDA extra: {cuda_extra}")

    conflicts: list[str] = []
    missing: list[str] = []
    for normalized_name in sorted(tracked_names):
        pins = package_to_envs.get(normalized_name)
        if not pins:
            missing.append(tracked_names[normalized_name])
            continue
        if len(pins) > 1:
            detail = ", ".join(
                f"{pin} ({', '.join(sorted(envs))})"
                for pin, envs in sorted(pins.items())
            )
            conflicts.append(f"# CONFLICT {tracked_names[normalized_name]}: {detail}")
            continue
        print(next(iter(pins)))

    for line in conflicts:
        print(line)
    for package_name in missing:
        print(f"# MISSING {package_name}")
    print("::endgroup::")


def print_untracked_direct_requirements(
    *, requirements_file: Path, constraints_file: Path
) -> None:
    tracked_names = set(parse_constraint_names(constraints_file))
    direct_names = parse_requirement_names(requirements_file)
    untracked_names = sorted(direct_names - tracked_names)

    print("::group::Untracked direct Python example requirements")
    print(f"# Requirements source: {requirements_file}")
    print(f"# Constraints source: {constraints_file}")
    if untracked_names:
        for name in untracked_names:
            print(name)
    else:
        print("# None")
    print("::endgroup::")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Format floating Python example dependency results for CI logs."
    )
    parser.add_argument("--constraints-file", type=Path, required=True)
    parser.add_argument("--requirements-file", type=Path, required=True)
    parser.add_argument("--freeze-dir", type=Path, required=True)
    parser.add_argument("--python-version", required=True)
    parser.add_argument("--cuda-extra", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print_constraints_candidate(
        constraints_file=args.constraints_file,
        freeze_dir=args.freeze_dir,
        python_version=args.python_version,
        cuda_extra=args.cuda_extra,
    )
    print_untracked_direct_requirements(
        requirements_file=args.requirements_file,
        constraints_file=args.constraints_file,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
