# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from cuda.bench.results import BenchmarkResult, BenchmarkResultSummary
from tabulate import tabulate

TILE_SHAPES = ("4x32", "8x16", "16x16", "32x8", "16x8", "8x8")
BENCHMARK_NAME = "stencil_autotune"
MEDIAN_TIE_RELATIVE_TOLERANCE = 0.01
MIN_RECOMMENDED_INTERIOR_PIXELS = 1_000_000


def parse_tile_shape(tile_shape: str) -> tuple[int, int]:
    block_x, block_y = tile_shape.split("x", maxsplit=1)
    return int(block_x), int(block_y)


def format_duration(seconds: float) -> str:
    if seconds >= 1.0:
        return f"{seconds:.3f} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.3f} ms"
    if seconds >= 1e-6:
        return f"{seconds * 1e6:.3f} us"
    return f"{seconds * 1e9:.3f} ns"


def format_optional_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    return format_duration(seconds)


def format_byte_rate(summary: BenchmarkResultSummary | None) -> str:
    if summary is None or summary.value is None:
        return "-"

    bytes_per_second = float(summary.value)
    if bytes_per_second >= 1e12:
        return f"{bytes_per_second * 1e-12:.3f} TB/s"
    if bytes_per_second >= 1e9:
        return f"{bytes_per_second * 1e-9:.3f} GB/s"
    if bytes_per_second >= 1e6:
        return f"{bytes_per_second * 1e-6:.3f} MB/s"
    if bytes_per_second >= 1e3:
        return f"{bytes_per_second * 1e-3:.3f} KB/s"
    return f"{bytes_per_second:.3f} B/s"


def state_tile_shape(state_name: str) -> str:
    prefix = "TileShape="
    for field in state_name.split():
        if field.startswith(prefix):
            return field.removeprefix(prefix)
    return state_name


def interior_pixel_count(width: int, height: int) -> int:
    return max(width - 2, 0) * max(height - 2, 0)


def median_ties_best(row: dict[str, Any], best_median_seconds: float) -> bool:
    tolerance = abs(best_median_seconds) * MEDIAN_TIE_RELATIVE_TOLERANCE
    return abs(row["median_seconds"] - best_median_seconds) <= tolerance


def summarize_result(result: BenchmarkResult) -> list[dict[str, Any]]:
    subbenchmark = result[BENCHMARK_NAME]
    medians = subbenchmark.centers(statistics.median)
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    rows = []

    for state in subbenchmark:
        median_seconds = medians[state.name()]
        if median_seconds is None:
            continue

        bandwidth = state.summaries.get("nv/cold/bw/global/bytes_per_second")
        mean_summary = state.summaries.get("nv/cold/time/gpu/mean")
        mean_seconds = (
            None
            if mean_summary is None or mean_summary.value is None
            else float(mean_summary.value)
        )
        rows.append(
            {
                "tile_shape": state_tile_shape(state.name()),
                "median_seconds": median_seconds,
                "mean_seconds": mean_seconds,
                "sample_count": len(state.samples) if state.samples is not None else 0,
                "bandwidth": format_byte_rate(bandwidth),
                "subprocess_seconds": metadata.get("elapsed_seconds", 0.0),
            }
        )

    return sorted(rows, key=lambda row: row["median_seconds"])


def print_summary(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("No benchmark states with sample data were found.")

    total_subprocess_seconds = sum(row["subprocess_seconds"] for row in rows)
    print()
    print(f"Total benchmark subprocess wall time: {total_subprocess_seconds:.3f} s")
    print()

    best_median_seconds = rows[0]["median_seconds"]
    tied_rows = [row for row in rows if median_ties_best(row, best_median_seconds)]
    table = [
        [
            "*" if row in tied_rows else "",
            row["tile_shape"],
            format_duration(row["median_seconds"]),
            format_optional_duration(row["mean_seconds"]),
            row["sample_count"],
            row["bandwidth"],
            f"{row['subprocess_seconds']:.3f} s",
        ]
        for row in rows
    ]
    print(
        tabulate(
            table,
            headers=[
                "Best",
                "TileShape",
                "Median GPU Time",
                "Mean GPU Time",
                "Samples",
                "GlobalMem BW",
                "Subprocess",
            ],
            tablefmt="simple",
            disable_numparse=True,
        )
    )

    print()
    if len(tied_rows) == 1:
        best = tied_rows[0]
        print(
            "Best tile shape by median isolated GPU time: "
            f"{best['tile_shape']} ({format_duration(best['median_seconds'])})"
        )
    else:
        tile_shapes = ", ".join(row["tile_shape"] for row in tied_rows)
        print(
            "No unique best tile shape by median isolated GPU time: "
            f"{len(tied_rows)} states are within "
            f"{MEDIAN_TIE_RELATIVE_TOLERANCE:.1%} of "
            f"{format_duration(best_median_seconds)} ({tile_shapes})."
        )


def run_driver(args: argparse.Namespace, nvbench_args: list[str]) -> int:
    with tempfile.TemporaryDirectory(prefix="nvbench-autotune-") as tmp_dir:
        rows = []
        total = len(TILE_SHAPES)
        interior_pixels = interior_pixel_count(args.image_width, args.image_height)
        print(
            f"Image size: {args.image_width}x{args.image_height} "
            f"({interior_pixels} interior stencil points)"
        )
        print(f"Sampling {total} tile shapes:")
        if interior_pixels < MIN_RECOMMENDED_INTERIOR_PIXELS:
            print(
                "Warning: this problem has only "
                f"{interior_pixels} interior stencil points. "
                "Small problems are usually dominated by kernel launch overhead, "
                "so median timings may tie across tile shapes."
            )

        for index, tile_shape in enumerate(TILE_SHAPES, start=1):
            json_path = Path(tmp_dir) / f"stencil_autotune_{tile_shape}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--run-benchmark",
                "--stopping-criterion",
                "entropy",
                "--tile-shape",
                tile_shape,
                "--image-width",
                str(args.image_width),
                "--image-height",
                str(args.image_height),
                "--jsonbin",
                str(json_path),
            ]
            if nvbench_args:
                command.extend(["--", *nvbench_args])

            print(f"[{index}/{total}] TileShape={tile_shape} ... ", end="", flush=True)
            start = time.perf_counter()
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            elapsed_seconds = time.perf_counter() - start

            if completed.returncode != 0:
                print(f"failed after {elapsed_seconds:.3f} s")
                print(completed.stdout, end="")
                return completed.returncode

            metadata = {
                "command": command,
                "returncode": completed.returncode,
                "elapsed_seconds": elapsed_seconds,
                "tile_shape": tile_shape,
            }
            result = BenchmarkResult.from_json(json_path, metadata=metadata)
            tile_rows = summarize_result(result)
            rows.extend(tile_rows)

            if tile_rows:
                row = tile_rows[0]
                print(
                    f"done in {elapsed_seconds:.3f} s, "
                    f"median {format_duration(row['median_seconds'])}, "
                    f"{row['bandwidth']}"
                )
            else:
                print(f"done in {elapsed_seconds:.3f} s, no samples")

        print_summary(sorted(rows, key=lambda row: row["median_seconds"]))
        return 0


def run_benchmark(args: argparse.Namespace, nvbench_args: list[str]) -> None:
    import cuda.bench as bench
    import numpy as np
    from numba import cuda

    def as_cuda_stream(cs: bench.CudaStream) -> cuda.cudadrv.driver.Stream:
        return cuda.external_stream(cs.addressof())

    @cuda.jit
    def stencil_kernel(inp, out, width, height):
        x, y = cuda.grid(2)
        if 0 < x < width - 1 and 0 < y < height - 1:
            idx = y * width + x
            out[idx] = 0.2 * (
                inp[idx]
                + inp[idx - 1]
                + inp[idx + 1]
                + inp[idx - width]
                + inp[idx + width]
            )

    def stencil_autotune(state: bench.State) -> None:
        tile_shape = state.get_string("TileShape")
        block_x, block_y = parse_tile_shape(tile_shape)
        width = args.image_width
        height = args.image_height
        interior_pixels = (width - 2) * (height - 2)

        state.add_element_count(interior_pixels, column_name="Pixels")
        state.add_global_memory_reads(
            interior_pixels * 5 * np.dtype(np.float32).itemsize
        )
        state.add_global_memory_writes(interior_pixels * np.dtype(np.float32).itemsize)

        host_input = np.ones(width * height, dtype=np.float32)
        dev_input = cuda.to_device(host_input)
        dev_output = cuda.device_array_like(dev_input)

        block_shape = (block_x, block_y)
        grid_shape = (
            (width + block_x - 1) // block_x,
            (height + block_y - 1) // block_y,
        )

        # Compile the Numba kernel outside NVBench measurement.
        stencil_kernel[grid_shape, block_shape](dev_input, dev_output, width, height)
        cuda.synchronize()

        def launcher(launch: bench.Launch) -> None:
            stream = as_cuda_stream(launch.get_stream())
            stencil_kernel[grid_shape, block_shape, stream, 0](
                dev_input,
                dev_output,
                width,
                height,
            )

        state.exec(launcher)

    benchmark = bench.register(stencil_autotune)
    benchmark.set_name(BENCHMARK_NAME)
    tile_shapes = [args.tile_shape] if args.tile_shape is not None else TILE_SHAPES
    benchmark.add_string_axis("TileShape", tile_shapes)
    bench.run_all_benchmarks([sys.argv[0], *nvbench_args])


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Autotune a simple stencil benchmark and select the best state "
            "from NVBench JSON-bin output."
        ),
        epilog=(
            "Additional NVBench options may be passed after '--'. "
            "For example: benchmark_result_autotune.py -- --timeout 30"
        ),
    )
    parser.add_argument(
        "--run-benchmark",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--tile-shape",
        choices=TILE_SHAPES,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=4096,
        help="Stencil input width used by the subprocess benchmark.",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=4096,
        help="Stencil input height used by the subprocess benchmark.",
    )
    args, nvbench_args = parser.parse_known_args(argv)
    if args.image_width < 3 or args.image_height < 3:
        parser.error("--image-width and --image-height must both be at least 3")
    nvbench_args = [arg for arg in nvbench_args if arg != "--"]
    return args, nvbench_args


def main(argv: list[str] | None = None) -> int:
    args, nvbench_args = parse_args(argv)
    if args.run_benchmark:
        run_benchmark(args, nvbench_args)
        return 0
    return run_driver(args, nvbench_args)


if __name__ == "__main__":
    sys.exit(main())
