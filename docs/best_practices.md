# NVBench Best Practices

NVBench is a **small but actively developed benchmarking library** for **CUDA GPU workloads**. It is **well-documented** and comes with many examples to help users get started quickly.

* [README](https://github.com/NVIDIA/cuCollections/blob/dev/README.md) — installation and basic usage.
* [Benchmark documentation](https://github.com/NVIDIA/nvbench/blob/main/docs/benchmarks.md) — detailed features.
* [Examples](https://github.com/NVIDIA/nvbench/tree/main/examples) — sample benchmarks.
* [CLI guides](https://github.com/NVIDIA/nvbench/blob/main/docs/cli_help.md) and [CLI axis documentation](https://github.com/NVIDIA/nvbench/blob/main/docs/cli_help_axis.md).

> **Note:** This document complements the official guides. All code is for demonstration purposes and is **not a production recommendation**.

---

## Key Features

* Purpose-built for **CUDA GPU workloads**.
* Provides **GPU-aware features**: warmup runs, synchronization, throughput/latency metrics, parameter sweeps, etc.
* Produces **machine-readable output** (JSON, CSV) for regression tracking and CI pipelines.
* **Natural choice for GPU benchmarking**, also supports CPU code.
* Python bindings are planned for future releases.

---

## Getting Started: Benchmarking a Simple GPU Kernel

### Naive Example

```cpp
void sequence_bench(nvbench::state& state) {
  auto data = thrust::device_vector<int>(10);
  state.exec([](nvbench::launch& launch) {
    thrust::sequence(data.begin(), data.end());
  });
}
NVBENCH_BENCH(sequence_bench);
```

> This may compile with **unused parameter warnings** and may **hang at runtime**, because NVBench requires explicit CUDA stream targeting and careful handling of synchronous kernels.

---

### Correct Usage with Stream

```cpp
void sequence_bench(nvbench::state& state) {
  auto data = thrust::device_vector<int>(10);
  state.exec([](nvbench::launch& launch) {
    thrust::sequence(thrust::device.on(launch.get_stream()), data.begin(), data.end());
  });
}
NVBENCH_BENCH(sequence_bench);
```

---

### Avoiding Deadlocks with `exec_tag::sync`

```cpp
void sequence_bench(nvbench::state& state) {
  auto data = thrust::device_vector<int>(10);
  state.exec(nvbench::exec_tag::sync, [&data](nvbench::launch& launch) {
    thrust::sequence(thrust::device.on(launch.get_stream()), data.begin(), data.end());
  });
}
NVBENCH_BENCH(sequence_bench);
```

> This ensures correct timing and avoids hangs caused by implicit synchronization in `thrust` calls.

---

## Multi-GPU Awareness

By default, NVBench runs on **all available GPUs**, which may increase runtime significantly. Target a specific GPU using:

```bash
export CUDA_VISIBLE_DEVICES=0
```

Example run output for a single GPU:

```bash
user@nvbench-test:~/nvbench/build/bin$ ./sequence_bench 
# Devices

## [0] `Quadro RTX 8000`
* SM Version: 750 (PTX Version: 750)
* Number of SMs: 72
* SM Default Clock Rate: 1770 MHz
* Global Memory: 48232 MiB Free / 48403 MiB Total
* Global Memory Bus Peak: 672 GB/sec (384-bit DDR @7001MHz)
* Max Shared Memory: 64 KiB/SM, 48 KiB/Block
* L2 Cache Size: 6144 KiB
* Maximum Active Blocks: 16/SM
* Maximum Active Threads: 1024/SM, 1024/Block
* Available Registers: 65536/SM, 65536/Block
* ECC Enabled: No

# Log

Run:  [1/1] sequence_bench [Device=0]
Pass: Cold: 0.006257ms GPU, 0.009850ms CPU, 0.50s total GPU, 4.40s total wall, 79920x 

# Benchmark Results

## sequence_bench

### [0] Quadro RTX 8000

| Samples | CPU Time | Noise | GPU Time | Noise  |
|---------|----------|-------|----------|--------|
|  79920x | 9.850 us | 9.62% | 6.257 us | 13.32% |
```

---

## Benchmarking Multiple Problem Sizes

Add an **axis** to test multiple input sizes without recompiling:

```cpp
void sequence_bench(nvbench::state& state) {
  auto const n = state.get_int64("Num");
  auto data = thrust::device_vector<int>(n);

  state.exec(nvbench::exec_tag::sync, [&data](nvbench::launch& launch) {
    thrust::sequence(thrust::device.on(launch.get_stream()), data.begin(), data.end());
  });
}

NVBENCH_BENCH(sequence_bench)
  .add_int64_axis("Num", {10, 100, 1000, 1000000});
```

CLI override example:

```bash
user@nvbench-test:~/nvbench/build/bin$ ./sequence_bench -a Num=[10,100000]
# Devices

## [0] `Quadro RTX 8000`
* SM Version: 750 (PTX Version: 750)
* Number of SMs: 72
* SM Default Clock Rate: 1770 MHz
* Global Memory: 48232 MiB Free / 48403 MiB Total
* Global Memory Bus Peak: 672 GB/sec (384-bit DDR @7001MHz)
* Max Shared Memory: 64 KiB/SM, 48 KiB/Block
* L2 Cache Size: 6144 KiB
* Maximum Active Blocks: 16/SM
* Maximum Active Threads: 1024/SM, 1024/Block
* Available Registers: 65536/SM, 65536/Block
* ECC Enabled: No

# Log

Run:  [1/2] sequence_bench [Device=0 Num=10]
Pass: Cold: 0.006318ms GPU, 0.009948ms CPU, 0.50s total GPU, 4.37s total wall, 79152x 
Run:  [2/2] sequence_bench [Device=0 Num=100000]
Pass: Cold: 0.006586ms GPU, 0.010193ms CPU, 0.50s total GPU, 4.14s total wall, 75936x 

# Benchmark Results

## sequence_bench

### [0] Quadro RTX 8000

|  Num   | Samples | CPU Time  | Noise | GPU Time | Noise  |
|--------|---------|-----------|-------|----------|--------|
|     10 |  79152x |  9.948 us | 9.63% | 6.318 us | 13.73% |
| 100000 |  75936x | 10.193 us | 9.62% | 6.586 us | 12.86% |
```

---

## Comparing Algorithms

You can easily benchmark alternative implementations. For example, replacing `thrust::sequence` with `thrust::transform`:

```cpp
void sequence_bench(nvbench::state& state) {
  auto const n = state.get_int64("Num");
  auto data = thrust::device_vector<int>(n);

  state.exec(nvbench::exec_tag::sync, [&data, n](nvbench::launch& launch) {
    thrust::transform(
      thrust::device.on(launch.get_stream()),
      thrust::counting_iterator<int>(0),
      thrust::counting_iterator<int>(n),
      data.begin(),
      cuda::std::identity{}
    );
  });
}
```

Record results to JSON for post-processing:

```bash
user@nvbench-test:~/nvbench/build/bin$ ./sequence_bench --json sequence_transform.json
```

Compare with reference `thrust::sequence` run using `nvbench_compare.py`:

```bash
user@nvbench-test:~/nvbench/scripts$ ./nvbench_compare.py sequence_ref.json sequence_transform.json 
['sequence_ref.json', 'sequence_transform.json']
# sequence_bench

## [0] Quadro RTX 8000

|   Num   |   Ref Time |   Ref Noise |   Cmp Time |   Cmp Noise |      Diff |   %Diff |  Status  |
|---------|------------|-------------|------------|-------------|-----------|---------|----------|
|   10    |   6.288 us |      13.70% |   6.301 us |      14.38% |  0.013 us |   0.20% |   SAME   |
|   100   |   6.331 us |      13.74% |   6.350 us |      15.15% |  0.019 us |   0.31% |   SAME   |
|  1000   |   6.548 us |      13.29% |   6.504 us |      13.95% | -0.043 us |  -0.66% |   SAME   |
| 1000000 |  12.528 us |       7.56% |  12.507 us |       8.41% | -0.021 us |  -0.17% |   SAME   |

# Summary

- Total Matches: 4  
  - Pass    (diff <= min_noise): 4  
  - Unknown (infinite noise):    0  
  - Failure (diff > min_noise):  0
```

> The two implementations perform nearly identically, demonstrating how NVBench can be used to **compare different algorithms or kernel implementations**.

---

## Summary

* Always **specify the CUDA stream** and use `exec_tag::sync` for synchronous kernels.
* Use **axes** and **CLI overrides** for flexible multi-size benchmarking.
* Record results in **JSON/CSV** for CI integration and regression analysis.
* NVBench is **actively developed**, easy to use, and ideal for **GPU benchmarking**, but note that it is small and has **limited community support**.

For more details and advanced examples, visit the [NVBench repository](https://github.com/NVIDIA/nvbench).
