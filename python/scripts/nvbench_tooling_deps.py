#!/usr/bin/env python
#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType


@dataclass(frozen=True)
class ToolingDependency:
    import_name: str
    package_name: str
    purpose: str
    extra: str = "tools"


class MissingToolingDependencyError(RuntimeError):
    pass


def require_tooling_dependency(
    dependency: ToolingDependency, *, tool_name: str
) -> ModuleType:
    try:
        return importlib.import_module(dependency.import_name)
    except ModuleNotFoundError as exc:
        top_level_package = dependency.import_name.partition(".")[0]
        if exc.name != top_level_package:
            raise
        raise MissingToolingDependencyError(
            f"{tool_name} requires {dependency.package_name!r} for "
            f"{dependency.purpose}.\n\n"
            "Install the required tooling dependencies with:\n"
            f"  python -m pip install 'cuda-bench[{dependency.extra}]'\n\n"
            f"Original import error: {exc}"
        ) from exc
