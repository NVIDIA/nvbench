<div align="center">

# NVBench — CUDA Kernel Benchmarking Library

**Measure CUDA kernel performance with confidence: sweep your kernel's
configuration space, gather accurate GPU timings, and get results you can
trust — with just a few lines of C++17 or Python.**

[![License](https://img.shields.io/badge/License-Apache%202.0%20with%20LLVM%20Exception-blue.svg)](./LICENSE)
[![cuda-bench on PyPI](https://img.shields.io/pypi/v/cuda-bench.svg?label=cuda-bench)](https://pypi.org/project/cuda-bench/)
[![Build and Test](https://github.com/NVIDIA/nvbench/actions/workflows/pr.yml/badge.svg)](https://github.com/NVIDIA/nvbench/actions/workflows/pr.yml)
[![Requirements](https://img.shields.io/badge/Requirements-CMake_3.30.4%2B_%E2%80%A2_CUDA_12%2B-76B900.svg)](#requirements)

</div>

> [!NOTE]
> This project is a work-in-progress. Everything is subject to change.

<details>
<summary><b>Table of Contents</b></summary>

* [Why NVBench?](#why-nvbench)
* [Quick Start (C++)](#quick-start-c)
* [Parameter Sweeps](#parameter-sweeps)
* [Measurement Types](#measurement-types)
* [Sample Output](#sample-output)
* [Quick Start (Python)](#quick-start-python)
* [Using NVBench in Your Project](#using-nvbench-in-your-project)
* [Requirements](#requirements)
* [Documentation](#documentation)
* [Examples](#examples)
* [Contributing](#contributing)
* [License](#license)
* [Scope and Related Projects](#scope-and-related-projects)

</details>

# Why NVBench?

Benchmarking GPU code is deceptively tricky: device clocks fluctuate, caches
stay warm between runs, and run-to-run noise hides real regressions. NVBench is
a C++17 library — with Python bindings — designed to simplify CUDA kernel
benchmarking and get it right for you.

* **[Parameter sweeps](docs/benchmarks.md#parameter-axes)** — a powerful and
  flexible "axis" system explores a kernel's configuration space. Parameters
  may be dynamic numbers/strings or
  [static types](docs/benchmarks.md#type-axes).
* **[Runtime customization](docs/cli_help.md)** — a rich command-line interface
  allows [redefinition of parameter axes](docs/cli_help_axis.md) without
  recompiling, CUDA device selection, locking GPU clocks (Volta+), changing
  output formats, and more.
* **[Throughput calculations](docs/benchmarks.md#throughput-measurements)** —
  compute and report item throughput (elements/second) and global memory
  bandwidth usage (bytes/second and per-device %-of-peak-bw).
* **[Hardware metrics](examples/auto_throughput.cu)** — collect hardware
  counters such as DRAM throughput, cache hit rates, and load/store efficiency
  with an opt-in CUPTI dependency (`NVBench_ENABLE_CUPTI=ON`).
* **[Configurable stopping criteria](examples/custom_criterion.cu)** — keep
  sampling until results are stable (`entropy`), stop at a fixed sample count,
  or register your own criterion.
* **Multiple output formats** — Markdown (default), CSV, and JSON.
* **[Manual timer mode](docs/benchmarks.md#explicit-timer-mode-nvbenchexec_tagtimer)**
  (optional) — explicitly start/stop timing in a benchmark implementation.
* **[Python bindings](python/README.md)** — benchmark kernels written in
  Python (Numba, CuPy, `cuda.core`, ...) with the `cuda-bench` package.

Check out [this talk](https://www.youtube.com/watch?v=CtrqBmYtSEk) for an
overview of the challenges inherent to CUDA kernel benchmarking and how NVBench
solves them for you!

# Quick Start (C++)

A basic kernel benchmark can be created with just a few lines of CUDA C++:

```cpp
// my_benchmark.cu
#include <nvbench/nvbench.cuh>

void my_benchmark(nvbench::state &state) {
  state.exec([](nvbench::launch &launch) {
    my_kernel<<<1024, 256, 0, launch.get_stream()>>>();
  });
}
NVBENCH_BENCH(my_benchmark);
```

Link it against `nvbench::main` in your `CMakeLists.txt`:

```cmake
include(FetchContent)
FetchContent_Declare(
  nvbench
  GIT_REPOSITORY https://github.com/NVIDIA/nvbench
  GIT_TAG        main
)
FetchContent_MakeAvailable(nvbench)

add_executable(my_bench my_benchmark.cu)
target_link_libraries(my_bench PRIVATE nvbench::main)
```

Build and run:

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build
./build/my_bench
```

Every NVBench executable also ships with a powerful CLI — run `--help` to see
all options for reconfiguring benchmarks, devices, and output at runtime. See
the [CLI reference](docs/cli_help.md) for details.

# Parameter Sweeps

NVBench automatically sweeps across any number of parameter axes. Here's a
benchmark that varies the number of inputs with a single line:

```cpp
void benchmark(nvbench::state &state) {
  const auto num_inputs = state.get_int64("NumInputs");
  thrust::device_vector<int> data = generate_input(num_inputs);

  state.exec([&data](nvbench::launch &launch) {
    my_kernel<<<blocks, threads, 0, launch.get_stream()>>>(
      thrust::raw_pointer_cast(data.data()), num_inputs);
  });
}
NVBENCH_BENCH(benchmark).add_int64_axis("NumInputs", {16, 64, 256, 1024, 4096});
```

Or with power-of-two ranges so the output stays readable:

```cpp
NVBENCH_BENCH(benchmark).add_int64_power_of_two_axis("NumInputs",
                                                     nvbench::range(4, 13, 2));
```

Combine multiple axes — including
[type axes](docs/benchmarks.md#type-axes) that instantiate kernel templates —
and NVBench explores the full Cartesian product automatically. See the
[parameter axes docs](docs/benchmarks.md#parameter-axes), the
[`axes` example](examples/axes.cu), and be careful with the
[combinatorial explosion lurking](docs/benchmarks.md#beware-combinatorial-explosion-is-lurking)
in multi-axis sweeps.

# Measurement Types

NVBench's timing methodology is designed so you always measure what you intend
to measure — be it cold-cache GPU performance, a thermal steady-state average,
or host-side CPU work:

* **Cold measurements**
  * Each sample runs the benchmark once with a clean device L2 cache.
  * GPU and CPU times are reported, along with noise estimates.
* **Batch measurements**
  * Executes the benchmark multiple times back-to-back and records total time.
  * Reports the average execution time (total time / number of executions).
* **[CPU-only measurements](docs/benchmarks.md#cpu-only-benchmarks)**
  * Measures the host-side execution time of a non-GPU benchmark.
  * Not suitable for microbenchmarking.

NVBench keeps sampling until the noise estimate converges below your target
(`--max-noise`) or the timeout expires — so you never wait on an unbounded
warm-up loop.

# Sample Output

<details>
  <summary>Full output from <code>nvbench.example.throughput</code></summary>

```
# Command Line

./bin/nvbench.example.throughput

# Devices

## [0] `Quadro GV100`
* SM Version: 700 (PTX Version: 700)
* Number of SMs: 80
* SM Default Clock Rate: 1627 MHz
* Global Memory: 32163 MiB Free / 32508 MiB Total
* Global Memory Bus Peak: 870 GiB/sec (4096-bit DDR @850MHz)
* Max Shared Memory: 96 KiB/SM, 48 KiB/Block
* L2 Cache Size: 6144 KiB
* Maximum Active Blocks: 32/SM
* Maximum Active Threads: 2048/SM, 1024/Block
* Available Registers: 65536/SM, 65536/Block
* ECC Enabled: No

# Log

Run:  throughput_bench [Device=0]
Warn: Current measurement timed out (15.00s) while over noise threshold (1.26% > 0.50%)
Pass: Cold: 0.262392ms GPU, 0.267860ms CPU, 7.19s total GPU, 27393x
Pass: Batch: 0.261963ms GPU, 7.18s total GPU, 27394x

# Benchmark Results

## throughput_bench

### [0] Quadro GV100

| NumElements |  DataSize  | Samples |  CPU Time  | Noise |  GPU Time  | Noise | Elem/s  | GlobalMem BW  | BWPeak | Batch GPU  | Batch  |
|-------------|------------|---------|------------|-------|------------|-------|---------|---------------|--------|------------|--------|
|    16777216 | 64.000 MiB |  27393x | 267.860 us | 1.25% | 262.392 us | 1.26% | 63.940G | 476.387 GiB/s | 58.77% | 261.963 us | 27394x |
```

</details>

# Quick Start (Python)

The same benchmarking power is available from Python with the
[`cuda-bench`](https://pypi.org/project/cuda-bench/) package — including
kernels authored with [Numba](python/examples/throughput.py),
[CuPy](python/examples/cupy_extract.py), [cuda.core](python/examples/axes.py),
and [cuda.compute](python/examples/cccl_compute_segmented_reduce.py):

```python
# my_bench.py
import sys

import cuda.bench as bench


@bench.register()
def my_benchmark(state: bench.State) -> None:
    state.exec(lambda launch: None)  # swap in your kernel launch


if __name__ == "__main__":
    bench.run_all_benchmarks(sys.argv)
```

```bash
python -m pip install cuda-bench      # or cuda-bench[cu12] / cuda-bench[cu13]
python my_bench.py
```

The published Linux wheels work with both CUDA 12.x and CUDA 13.x Python
environments. See the [Python package README](python/README.md) for
installation details and a full API guide.

# Using NVBench in Your Project

To try NVBench with your own kernels without much investment, clone the
[NVBench Demo Project](https://github.com/allisonvacanti/nvbench_demo) —
a minimal CMake project ready to build your first benchmark.

If you prefer managing dependencies manually, the essentials are:

```cmake
include(cmake/CPM.cmake)                  # https://github.com/cpm-cmake/CPM.cmake
CPMAddPackage("gh:NVIDIA/nvbench")

target_link_libraries(my_executable PRIVATE nvbench::main)
```

Useful CMake options:

| Option | Description |
|--------|-------------|
| `NVBench_ENABLE_EXAMPLES` | Build the example benchmarks (default: `OFF`) |
| `NVBench_ENABLE_TESTING` | Build the NVBench test suite (default: `OFF`) |
| `NVBench_ENABLE_CUPTI` | Enable hardware metric collection through CUPTI |
| `NVBench_ENABLE_NVML` | Build with NVML support from the CUDA Toolkit |
| `NVBench_ENABLE_EXECUTABLES` | Build NVBench command-line executables |
| `NVBench_ENABLE_INSTALL_RULES` | Add NVBench install rules |

# Requirements

| Requirement | Version |
|-------------|---------|
| CMake | 3.30.4+ |
| CUDA Toolkit + `nvcc` | 12.0 and above |
| g++ | 7 – 14 |
| clang++ | 14 – 19 |

Headers are tested with C++17 through C++20.

# Documentation

| Resource | Description |
|----------|-------------|
| [Benchmark guide](docs/benchmarks.md) | Everything you can do in a benchmark: parameter axes, streams, tags, skipping, CPU-only mode, and more |
| [CLI reference](docs/cli_help.md) | All command-line options, with [argument details for redefining axes](docs/cli_help_axis.md) |
| [Comparing results](docs/nvbench_compare_robust.md) | `nvbench-compare-robust`: conservatively compare two runs and report `SAME`/`FAST`/`SLOW` |
| [Legacy data migration](docs/nvbench_normalize_jsonbin.md) | `nvbench-normalize-jsonbin`: migrate results produced by older NVBench versions |
| [Python API](python/README.md) | The `cuda-bench` package: installation, API walkthrough, and Python examples |
| [Examples](examples/) | Ready-to-build C++ samples demonstrating every major feature, including [custom stopping criteria](examples/custom_criterion.cu) and [CUPTI metrics](examples/auto_throughput.cu) |
| [Demo project](https://github.com/allisonvacanti/nvbench_demo) | A minimal starter project for benchmarking your own kernels |
| [Conference talk](https://www.youtube.com/watch?v=CtrqBmYtSEk) | Video overview of the challenges inherent to CUDA kernel benchmarking, and how NVBench solves them |

# Examples

This repository provides a number of
[examples](examples/) that demonstrate various NVBench features and use cases:

| Example | Demonstrates |
|---------|--------------|
| [axes.cu](examples/axes.cu) | Runtime and compile-time parameter sweeps |
| [enums.cu](examples/enums.cu) | Enums and compile-time-constant-integral parameter axes |
| [throughput.cu](examples/throughput.cu) | Reporting item/sec and byte/sec throughput statistics |
| [auto_throughput.cu](examples/auto_throughput.cu) | Collecting hardware metrics, such as DRAM throughput and cache hit rates (requires CUPTI) |
| [cpu_only.cu](examples/cpu_only.cu) | CPU-only benchmarking |
| [custom_criterion.cu](examples/custom_criterion.cu) | Defining custom stopping criteria |
| [skip.cu](examples/skip.cu) | Skipping benchmark configurations |
| [stream.cu](examples/stream.cu) | Benchmarking on a specific CUDA stream |
| [summaries.cu](examples/summaries.cu) | Adding / hiding columns (summaries) in markdown output |
| [exec_tag_sync.cu](examples/exec_tag_sync.cu) | Benchmarks that synchronize CUDA devices: `nvbench::exec_tag::sync` |
| [exec_tag_timer.cu](examples/exec_tag_timer.cu) | Manual timing: `nvbench::exec_tag::timer` |

### Building Examples

To build the examples:

```bash
mkdir -p build
cd build
cmake -DNVBench_ENABLE_EXAMPLES=ON -DCMAKE_CUDA_ARCHITECTURES=70 .. && make
```

Set `CMAKE_CUDA_ARCHITECTURES` based on the GPU you are running on. Examples
are built by default into `build/bin` and are prefixed with `nvbench.example`.

# Contributing

Contributions are welcome!

For current issues, see the [issue
board](https://github.com/NVIDIA/nvbench/issues). Issues labeled with
[![good first issue](https://img.shields.io/github/labels/NVIDIA/nvbench/good%20first%20issue)](https://github.com/NVIDIA/nvbench/labels/good%20first%20issue)
are good for first-time contributors.

Community support is also available via
[GitHub Discussions](https://github.com/NVIDIA/nvbench/discussions).

## Tests

To build the `nvbench` tests:

```bash
mkdir -p build
cd build
cmake -DNVBench_ENABLE_TESTING=ON .. && make
```

Tests are built by default into `build/bin` and prefixed with `nvbench.test`.

To run all tests:

```bash
make test
```
or
```bash
ctest
```

# License

NVBench is released under the [Apache 2.0 License with LLVM
exceptions](./LICENSE).

# Scope and Related Projects

NVBench will measure the CPU and CUDA GPU execution time of a ***single
host-side critical region*** per benchmark. It is intended for regression
testing and parameter tuning of individual kernels. For in-depth analysis of
end-to-end performance of multiple applications, the NVIDIA
[Nsight Systems](https://developer.nvidia.com/nsight-systems) and
[Nsight Compute](https://developer.nvidia.com/nsight-compute) tools are more
appropriate.

NVBench is focused on evaluating the performance of CUDA kernels. It also
provides [CPU-only benchmarking
facilities](docs/benchmarks.md#cpu-only-benchmarks) intended for non-trivial
CPU workloads, but is not optimized for CPU microbenchmarks. This may change in
the future, but for now, consider using
[Google Benchmark](https://github.com/google/benchmark) for high resolution CPU
benchmarks.
