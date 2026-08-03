#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

QUIET_NVBENCH_ARG = "-q"


@dataclass(frozen=True)
class ExampleRun:
    name: str
    path: str
    groups: tuple[str, ...]
    required_modules: tuple[str, ...] = ()
    requires_gpu: bool = False
    use_profile: bool = True
    script_args: tuple[str, ...] = ()
    nvbench_args: tuple[str, ...] = ()
    nvbench_args_after_separator: bool = False
    timeout_seconds: int = 300
    description: str = ""


EXAMPLE_RUNS = (
    ExampleRun(
        name="python-smoke",
        path="python/test/smoke.py",
        groups=("smoke",),
        required_modules=("cuda.bench",),
        timeout_seconds=60,
        description="CPU-only cuda.bench smoke benchmark.",
    ),
    ExampleRun(
        name="cpu-activity-cpu-only",
        path="python/examples/cpu_activity.py",
        groups=("example-cpu",),
        required_modules=("cuda.bench", "cuda.cccl.headers", "cuda.core"),
        script_args=("-b", "cpu_only_sleep_bench"),
        timeout_seconds=120,
        description="CPU-only benchmark from a committed example file.",
    ),
    ExampleRun(
        name="axes-default-value",
        path="python/examples/axes.py",
        groups=("core-cccl",),
        required_modules=("cuda.bench", "cuda.cccl.headers", "cuda.core"),
        requires_gpu=True,
        script_args=("-b", "default_value"),
        description="CUDA Python example with no axis override.",
    ),
    ExampleRun(
        name="exec-tag-sync",
        path="python/examples/exec_tag_sync.py",
        groups=("core-cccl",),
        required_modules=("cuda.bench", "cuda.cccl.headers", "cuda.core"),
        requires_gpu=True,
        script_args=("-b", "synchronizing_bench"),
        description="exec_tag::sync example.",
    ),
    ExampleRun(
        name="exec-tag-timer",
        path="python/examples/exec_tag_timer.py",
        groups=("core-cccl",),
        required_modules=("cuda.bench", "cuda.cccl.headers", "cuda.core"),
        requires_gpu=True,
        script_args=("-b", "mod2_inplace"),
        description="exec_tag::timer example.",
    ),
    ExampleRun(
        name="skip",
        path="python/examples/skip.py",
        groups=("core-cccl",),
        required_modules=("cuda.bench", "cuda.cccl.headers", "cuda.core"),
        requires_gpu=True,
        script_args=(
            "-b",
            "runtime_skip",
            "-a",
            "Duration=0.0001",
            "-a",
            "Kramble=Foo",
        ),
        description="Runtime skip example with one concrete axis configuration.",
    ),
    ExampleRun(
        name="throughput",
        path="python/examples/throughput.py",
        groups=("numba",),
        required_modules=("cuda.bench", "numba", "numba.cuda", "numpy"),
        requires_gpu=True,
        script_args=(
            "-b",
            "throughput_bench",
            "-a",
            "Stride=1",
            "-a",
            "ItemsPerThread=1",
        ),
        description="Numba throughput example.",
    ),
    ExampleRun(
        name="cuda-coop-block-reduce",
        path="python/examples/cuda_coop_block_reduce.py",
        groups=("numba",),
        required_modules=("cuda.bench", "cuda.coop", "numba", "numpy"),
        requires_gpu=True,
        script_args=(
            "-b",
            "multi_block_bench",
            "-a",
            "ThreadsPerBlock=64",
            "-a",
            "NumBlocks[pow2]=10",
        ),
        description="cuda.coop block reduce example.",
    ),
    ExampleRun(
        name="cupy-extract",
        path="python/examples/cupy_extract.py",
        groups=("cupy",),
        required_modules=("cuda.bench", "cupy"),
        requires_gpu=True,
        script_args=(
            "-b",
            "cupy_extract_by_mask",
            "-a",
            "numCols=1024",
            "-a",
            "numRows=1024",
        ),
        description="CuPy external stream example.",
    ),
    ExampleRun(
        name="stream",
        path="python/examples/stream.py",
        groups=("cupy",),
        required_modules=("cuda.bench", "cupy"),
        requires_gpu=True,
        script_args=(
            "-b",
            "elementwise_square",
            "-a",
            "Elements[pow2]=22",
        ),
        description="State.set_stream example using a CuPy stream.",
    ),
    ExampleRun(
        name="cuda-compute-segmented-reduce",
        path="python/examples/cuda_compute_segmented_reduce.py",
        groups=("cuda-compute",),
        required_modules=(
            "cuda.bench",
            "cuda.compute.algorithms",
            "cuda.compute.iterators",
            "cuda.core",
            "cupy",
            "numpy",
        ),
        requires_gpu=True,
        script_args=(
            "-b",
            "segmented_reduce",
            "-a",
            "numElems=1048576",
            "-a",
            "numCols=1024",
        ),
        description="cuda.compute segmented reduce example.",
    ),
    ExampleRun(
        name="benchmark-result-autotune",
        path="python/examples/benchmark_result_autotune.py",
        groups=("autotune",),
        required_modules=(
            "cuda.bench",
            "cuda.bench.results",
            "numba",
            "numpy",
            "tabulate",
        ),
        requires_gpu=True,
        use_profile=False,
        script_args=(
            "--image-width",
            "64",
            "--image-height",
            "64",
        ),
        nvbench_args=(
            "--stopping-criterion",
            "sample-count",
            "--target-samples",
            "100",
        ),
        nvbench_args_after_separator=True,
        timeout_seconds=300,
        description="BenchmarkResult-driven autotune example with a small input.",
    ),
    ExampleRun(
        name="pytorch-bench",
        path="python/examples/pytorch_bench.py",
        groups=("torch",),
        required_modules=("cuda.bench", "torch"),
        requires_gpu=True,
        script_args=("-b", "torch_bench"),
        timeout_seconds=300,
        description="PyTorch external stream example.",
    ),
    ExampleRun(
        name="cute-dsl-sgemm",
        path="python/examples/cute_dsl_sgemm.py",
        groups=("cute",),
        required_modules=(
            "cuda.bench",
            "cuda.bindings.driver",
            "cuda.core",
            "cupy",
            "cutlass",
            "numpy",
        ),
        requires_gpu=True,
        script_args=("-b", "cutlass_gemm", "-a", "R=16", "-a", "N=256"),
        timeout_seconds=600,
        description="CUTLASS/CuTe DSL SGEMM example.",
    ),
)


GROUP_ALIASES = {
    "pr": ("syntax", "smoke"),
    "cpu": ("smoke", "example-cpu"),
    "light-gpu": ("core-cccl",),
    "gpu": ("core-cccl", "numba", "cupy", "cuda-compute"),
    "all": (
        "syntax",
        "smoke",
        "example-cpu",
        "core-cccl",
        "numba",
        "cupy",
        "cuda-compute",
        "autotune",
        "torch",
        "cute",
    ),
}


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_command(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: int,
    verbose: bool,
) -> tuple[bool, str, float]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        elapsed = time.perf_counter() - start
        output = e.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        return False, f"Timed out after {timeout_seconds}s.\n{output}", elapsed
    except OSError as e:
        elapsed = time.perf_counter() - start
        return False, f"Failed to run {shell_join(command)}: {e}", elapsed

    elapsed = time.perf_counter() - start
    output = completed.stdout or ""
    if verbose or completed.returncode != 0:
        return completed.returncode == 0, output, elapsed
    return completed.returncode == 0, "", elapsed


def tracked_example_scripts(repo_root: Path) -> list[Path]:
    command = ["git", "ls-files", "--", "python/examples"]
    try:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except OSError:
        completed = None

    if completed is not None and completed.returncode == 0:
        paths = [
            Path(line) for line in completed.stdout.splitlines() if line.endswith(".py")
        ]
        if paths:
            return sorted(paths)

    return sorted(
        path.relative_to(repo_root)
        for path in (repo_root / "python" / "examples").rglob("*.py")
    )


def find_missing_modules(
    python: str, modules: tuple[str, ...], repo_root: Path
) -> list[str]:
    if not modules:
        return []

    code = r"""
import importlib.util
import sys

missing = []
for module_name in sys.argv[1:]:
    try:
        spec = importlib.util.find_spec(module_name)
    except Exception:
        spec = None
    if spec is None:
        missing.append(module_name)

print("\n".join(missing))
raise SystemExit(1 if missing else 0)
"""
    try:
        completed = subprocess.run(
            [python, "-c", code, *modules],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except OSError as e:
        return [f"failed to run {python}: {e}"]
    return [line for line in completed.stdout.splitlines() if line]


def expand_groups(groups: list[str]) -> set[str]:
    expanded: set[str] = set()
    for group in groups:
        alias_groups = GROUP_ALIASES.get(group)
        if alias_groups is None:
            expanded.add(group)
        else:
            expanded.update(alias_groups)
    return expanded


def known_groups() -> list[str]:
    groups = {"syntax"}
    for run in EXAMPLE_RUNS:
        groups.update(run.groups)
    groups.update(GROUP_ALIASES)
    return sorted(groups)


def selected_runs(groups: set[str], case_names: list[str]) -> list[ExampleRun]:
    runs = [run for run in EXAMPLE_RUNS if groups.intersection(run.groups)]
    if not case_names:
        return runs

    names = set(case_names)
    matched_names = {run.name for run in runs if run.name in names}
    missing = sorted(names - matched_names)
    if missing:
        known = ", ".join(run.name for run in EXAMPLE_RUNS)
        raise RuntimeError(
            f"Unknown or unselected case(s): {', '.join(missing)}. Known cases: {known}"
        )
    return [run for run in runs if run.name in names]


def build_run_command(
    run: ExampleRun,
    *,
    python: str,
    device: str,
) -> list[str]:
    command = [python, run.path]
    command.extend(run.script_args)

    nvbench_args = [QUIET_NVBENCH_ARG]
    if run.use_profile:
        nvbench_args.append("--profile")
    if run.requires_gpu:
        nvbench_args.extend(("--devices", device))
    nvbench_args.extend(run.nvbench_args)

    if run.nvbench_args_after_separator:
        command.append("--")
    command.extend(nvbench_args)
    return command


def execute_and_report(
    name: str,
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: int,
    verbose: bool,
) -> bool:
    print(f"RUN {name}: {shell_join(command)}", flush=True)
    passed, output, elapsed = run_command(
        command,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        verbose=verbose,
    )
    status = "PASS" if passed else "FAIL"
    print(f"{status} {name} ({elapsed:.2f}s)")
    if output:
        print(output, end="" if output.endswith("\n") else "\n")
    return passed


def run_syntax_check(args: argparse.Namespace, repo_root: Path) -> bool:
    example_paths = tracked_example_scripts(repo_root)
    command = [
        args.python,
        "-m",
        "py_compile",
        *(str(path) for path in example_paths),
    ]
    return execute_and_report(
        "syntax",
        command,
        cwd=repo_root,
        timeout_seconds=args.timeout,
        verbose=args.verbose,
    )


def run_example_case(args: argparse.Namespace, repo_root: Path, run: ExampleRun) -> str:
    path = repo_root / run.path
    if not path.exists():
        print(f"FAIL {run.name}: missing file {run.path}")
        return "failed"

    missing_modules = find_missing_modules(args.python, run.required_modules, repo_root)
    if missing_modules:
        message = f"missing Python module(s): {', '.join(missing_modules)}"
        if args.skip_missing_deps:
            print(f"SKIP {run.name}: {message}")
            return "skipped"
        print(f"FAIL {run.name}: {message}")
        return "failed"

    command = build_run_command(run, python=args.python, device=args.device)
    passed = execute_and_report(
        run.name,
        command,
        cwd=repo_root,
        timeout_seconds=run.timeout_seconds,
        verbose=args.verbose,
    )
    return "passed" if passed else "failed"


def print_listing() -> None:
    print("Groups:")
    for group in known_groups():
        alias = GROUP_ALIASES.get(group)
        if alias is None:
            print(f"  {group}")
        else:
            print(f"  {group} -> {', '.join(alias)}")

    print("\nCases:")
    for run in EXAMPLE_RUNS:
        gpu = "gpu" if run.requires_gpu else "cpu"
        groups = ",".join(run.groups)
        print(f"  {run.name:32} {gpu:3} [{groups}] {run.path}")
        if run.description:
            print(f"    {run.description}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run syntax and smoke checks for committed Python examples."
    )
    parser.add_argument(
        "--group",
        action="append",
        choices=known_groups(),
        default=[],
        help=(
            "Group to run. May be specified multiple times. "
            "Defaults to 'syntax'. Use 'pr' for syntax plus CPU-only smoke."
        ),
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Run a selected case from the selected groups. May be repeated.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for checks and example subprocesses.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root_from_script(),
        help="Repository root. Defaults to the parent directory of this script.",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="NVBench device selector passed to GPU examples. Defaults to 0.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout in seconds for the syntax check.",
    )
    parser.add_argument(
        "--skip-missing-deps",
        action="store_true",
        help="Skip runtime cases whose required Python modules are not installed.",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue running later cases after a failure.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print stdout/stderr from passing commands.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List known groups and runtime cases, then exit.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    repo_root = args.repo_root.resolve()

    if args.list:
        print_listing()
        return 0

    groups = expand_groups(args.group or ["syntax"])
    failures = 0
    skipped = 0
    passed = 0

    if "syntax" in groups:
        if run_syntax_check(args, repo_root):
            passed += 1
        else:
            failures += 1
            if not args.continue_on_failure:
                return 1

    try:
        runs = selected_runs(groups, args.case)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    for run in runs:
        result = run_example_case(args, repo_root, run)
        if result == "passed":
            passed += 1
        elif result == "skipped":
            skipped += 1
        else:
            failures += 1
            if not args.continue_on_failure:
                return 1

    print(f"Summary: {passed} passed, {skipped} skipped, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
