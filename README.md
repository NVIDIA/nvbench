# Overview

This project is a work-in-progress. Everything is subject to change.

NVBench is a C++17 library designed to simplify CUDA kernel benchmarking. It
features:

* [Parameter sweeps](docs/benchmarks.md#parameter-axes): a powerful and
  flexible "axis" system explores a kernel's configuration space. Parameters may
  be dynamic numbers/strings or [static types](docs/benchmarks.md#type-axes).
* [Runtime customization](docs/cli_help.md): A rich command-line interface
  allows [redefinition of parameter axes](docs/cli_help_axis.md), CUDA device
  selection, changing output formats, and more.
* [Throughput calculations](docs/benchmarks.md#throughput-measurements): Compute
  and report:
  * Item throughput (elements/second)
  * Global memory bandwidth usage (bytes/second and per-device %-of-peak-bw)
* Multiple output formats: Currently supports markdown (default) and CSV output.
* [Manual timer mode](docs/benchmarks.md#explicit-timer-mode-nvbenchexec_tagtimer):
  (optional) Explicitly start/stop timing in a benchmark implementation.
* Multiple measurement types:
  * Cold Measurements:
    * Each sample runs the benchmark once with a clean device L2 cache.
    * GPU and CPU times are reported.
  * Batch Measurements:
    * Executes the benchmark multiple times back-to-back and records total time.
    * Reports the average execution time (total time / number of executions).

# Getting Started

## Minimal Benchmark

A basic kernel benchmark can be created with just a few lines of CUDA C++:

```cpp
void my_benchmark(nvbench::state& state) {
  state.exec([](nvbench::launch& launch) { 
    my_kernel<<<num_blocks, 256, 0, launch.get_stream()>>>();
  });
}
NVBENCH_BENCH(my_benchmark);
```

See [Benchmarks](docs/benchmarks.md) for information on customizing benchmarks
and implementing parameter sweeps.

## Command Line Interface

Each benchmark executable produced by NVBench provides a rich set of
command-line options for configuring benchmark execution at runtime. See the
[CLI overview](docs/cli_help.md)
and [CLI axis specification](docs/cli_help_axis.md) for more information.

## Examples

This repository provides a number of [examples](examples/) that demonstrate
various NVBench features and usecases:

- [Runtime and compile-time parameter sweeps](examples/axes.cu)
- [Enums and compile-time-constant-integral parameter axes](examples/enums.cu)
- [Reporting item/sec and byte/sec throughput statistics](examples/throughput.cu)
- [Skipping benchmark configurations](examples/skip.cu)
- [Benchmarks that sync CUDA devices: `nvbench::exec_tag::sync`](examples/exec_tag_sync.cu)
- [Manual timing: `nvbench::exec_tag::timer`](examples/exec_tag_timer.cu)

To get started using NVBench with your own kernels, consider trying out
the [NVBench Demo Project](https://github.com/allisonvacanti/nvbench_demo)
. `nvbench_demo` provides a simple CMake project that uses NVBench to build an
example benchmark. It's a great way to experiment with the library without a lot
of investment.

# License

NVBench is released under the Apache 2.0 License with LLVM exceptions.
See [LICENSE](./LICENSE).

# Scope and Related Projects

NVBench will measure the CPU and CUDA GPU execution time of a ***single
host-side critical region*** per benchmark. It is intended for regression
testing and parameter tuning of individual kernels. For in-depth analysis of
end-to-end performance of multiple applications, the NVIDIA Nsight tools are
more appropriate.

NVBench is focused on evaluating the performance of CUDA kernels and is not
optimized for CPU microbenchmarks. This may change in the future, but for now,
consider using Google Benchmark for high resolution CPU benchmarks.
