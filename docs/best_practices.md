# NVBench Best Practices

NVBench is a **small yet actively developed benchmarking library** for CUDA GPU workloads. The [README](https://github.com/NVIDIA/cuCollections/blob/dev/README.md) serves as the ideal starting point, providing detailed guidance for users to get hands-on experience—from installation to framework usage. It includes links to the [benchmark documentation](https://github.com/NVIDIA/nvbench/blob/main/docs/benchmarks.md), which covers all essential features and usage instructions, as well as links to [code examples](https://github.com/NVIDIA/nvbench/tree/main/examples) that demonstrate how to integrate NVBench into a user’s codebase.

This document is **not intended to replace** the detailed benchmark documentation ([here](https://github.com/NVIDIA/nvbench/blob/main/docs/benchmarks.md)) or the CLI help guides ([CLI help](https://github.com/NVIDIA/nvbench/blob/main/docs/cli_help.md) and [CLI axis help](https://github.com/NVIDIA/nvbench/blob/main/docs/cli_help_axis.md)). All examples provided are for demonstration purposes and are **not intended as recommendations for best practices in real-world scenarios**.

## NVBench

* Purpose-built for CUDA GPU workloads.
* Provides GPU-aware features: warmup runs, synchronization, throughput/latency metrics, and parameter sweeps, etc.
* Produces machine-readable output (JSON, CSV) suitable for regression tracking and CI pipelines.
* The natural choice for benchmarking GPU-accelerated code.
* Also supports benchmarking normal CPU implementations.
* Python code support is in the roadmap.

## Benchmark Your GPU Code with NVBench
Let’s begin with a simple example for users who are new to NVBench and want to learn the basics of benchmarking GPU code. Consider measuring the performance of `thrust::sequence` on a GPU. Similar to `std::iota`, suppose we have an input array of 10 elements, and we want `thrust::sequence` to populate it with the sequence of values from 0 to 9. The following example demonstrates this approach:

```cpp
void sequence_bench(nvbench::state& state) {
  auto data = thrust::device_vector<int>(10);
  state.exec([](nvbench::launch& launch) {
    thrust::sequence(data.begin(), data.end());
  });
}
NVBENCH_BENCH(sequence_bench);
```
Will this code work as-is? Depending on the build system configuration, compilation may succeed but generate warnings indicating that `launch` is an unused parameter. The code may or may not execute correctly. This often occurs when users, accustomed to a sequential programming mindset, overlook the fact that GPU architectures are highly parallel. Proper use of streams and synchronization is essential for accurately measuring performance in benchmark code.

A common mistake in this context is neglecting stream specification: NVBench requires knowledge of the exact CUDA stream being targeted to correctly trace kernel execution and measure performance. Therefore, users must explicitly provide the stream to be benchmarked. For example, passing the NVBench launch stream ensures correct execution and accurate measurement:

```cpp
void sequence_bench(nvbench::state& state) {
  auto data = thrust::device_vector<int>(10);
  state.exec([](nvbench::launch& launch) {
    thrust::sequence(thrust::device.on(launch.get_stream()), data.begin(), data.end());
  });
}
NVBENCH_BENCH(sequence_bench);
```

By explicitly specifying `launch.get_stream()`, NVBench can correctly target the kernels executed on that stream. After recompilation, the compilation warnings will be resolved, and the build will complete successfully. However, at runtime, the code may hang, for example:


```bash

######################################################################
##################### Possible Deadlock Detected #####################
######################################################################

Forcing unblock: The current measurement appears to have deadlocked
and the results cannot be trusted.

This happens when the KernelLauncher synchronizes the CUDA device.
If this is the case, pass the `sync` exec_tag to the `exec` call:

    state.exec(<KernelLauncher>); // Deadlock
    state.exec(nvbench::exec_tag::sync, <KernelLauncher>); // Safe
```

The runtime execution log indicates a deadlock, and NVBench terminated the run to prevent unnecessary execution. The log shows that the issue arises from implicit synchronization within the target kernel—in this case, the `thrust::sequence` call. By default, unless explicitly specified, `thrust` uses a synchronous execution policy internally. Therefore, users must pass `nvbench::exec_tag::sync` to ensure correct benchmarking. This will **not** produce a build-time error but can cause runtime hangs if omitted.

Now, we fix the code:

```cpp
void sequence_bench(nvbench::state& state) {
  auto data = thrust::device_vector<int>(10);
  state.exec(nvbench::exec_tag::sync, [&data](nvbench::launch& launch) {
    thrust::sequence(thrust::device.on(launch.get_stream()), data.begin(), data.end());
  });
}
NVBENCH_BENCH(sequence_bench);
```

When the benchmark is executed, results are displayed without issues. However, users, particularly in a multi-GPU environment, may observe that more results are collected than expected:

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

## [1] `NVIDIA RTX A400`
* SM Version: 860 (PTX Version: 860)
* Number of SMs: 6
* SM Default Clock Rate: 1762 MHz
* Global Memory: 2801 MiB Free / 3769 MiB Total
* Global Memory Bus Peak: 96 GB/sec (64-bit DDR @6001MHz)
* Max Shared Memory: 100 KiB/SM, 48 KiB/Block
* L2 Cache Size: 1024 KiB
* Maximum Active Blocks: 16/SM
* Maximum Active Threads: 1536/SM, 1024/Block
* Available Registers: 65536/SM, 65536/Block
* ECC Enabled: No

# Log

Run:  [1/2] sequence_bench [Device=0]
Pass: Cold: 0.006150ms GPU, 0.009768ms CPU, 0.50s total GPU, 4.52s total wall, 81312x 
Run:  [2/2] sequence_bench [Device=1]
Pass: Cold: 0.007819ms GPU, 0.013864ms CPU, 0.50s total GPU, 3.59s total wall, 63952x 

# Benchmark Results

## sequence_bench

### [0] Quadro RTX 8000

| Samples | CPU Time | Noise  | GPU Time | Noise  |
|---------|----------|--------|----------|--------|
|  81312x | 9.768 us | 13.55% | 6.150 us | 20.16% |

### [1] NVIDIA RTX A400

| Samples | CPU Time  |  Noise  | GPU Time |  Noise  |
|---------|-----------|---------|----------|---------|
|  63952x | 13.864 us | 432.95% | 7.819 us | 447.95% |
```

By default, NVBench runs all GPUs locally unless specified. If not specified, it will run all available GPUs. This is especially problematic if your system has multiple GPUs and you want to target a particular GPU to save build time. In our case, we target **RTX8000**:

```bash
user@nvbench-test:~/nvbench/build/bin$ export CUDA_VISIBLE_DEVICES=0
```

Now, if we rerun:

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

## Benchmarking Multiple Problem Sizes

Benchmarking the performance of a single problem size is usually **less desired in real-world problems**. In most cases, we want to run different problem sizes for the same kernel. NVBench provides an **“axis”** feature to help with this. For example, to test input sizes from `10` to `1000000`:

```cpp
void sequence_bench(nvbench::state& state) {
  auto const n = state.get_int64("Num");
  auto data = thrust::device_vector<int>(n);

  state.exec(nvbench::exec_tag::sync, [&data](nvbench::launch& launch) {
    thrust::sequence(thrust::device.on(launch.get_stream()), data.begin(), data.end());
  });
}
NVBENCH_BENCH(sequence_bench)
  .add_int64_axis("Num", std::vector<nvbench::int64_t>{10, 100, 1000, 1000000});
```

**Axis is a powerful tool** provided by NVBench. Users may encounter situations where they want to test only certain sizes. NVBench provides a **flexible CLI**, so users can change the benchmark parameters **without recompiling the code**:

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

For more details about **CLI axis control**, please check [here](https://github.com/NVIDIA/nvbench/blob/main/docs/cli_help_axis.md).

## Comparing Algorithms Using NVBench

Once benchmarks are set, a major use is to evaluate performance between different algorithms. For example, the same sequence algorithm can be written manually using `thrust::transform`. We can compare the performance of a manual transform sequence against `thrust::sequence`.

### Step 1: Record Reference Performance

Record the `thrust::sequence` benchmark in a JSON file for post-processing:

```bash
user@nvbench-test:~/nvbench/build/bin$ ./sequence_bench --json sequence_ref.json
```

### Step 2: Update Code with `thrust::transform`

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

### Step 3: Run Benchmark with Transform and Save JSON

```bash
user@nvbench-test:~/nvbench/build/bin$ ./sequence_bench --json sequence_transform.json
```

### Step 4: Compare Results

NVBench provides a convenient script under `nvbench/scripts` called `nvbench_compare.py`. After copying the JSON files to the scripts folder:

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

We can see that the performance of the two approaches is essentially the same.

---

For more information on how to use NVBench in your projects, please check the [NVBench repository](https://github.com/NVIDIA/nvbench). Feel free to raise questions or feature requests via **GitHub issues** or **discussions**, and enjoy benchmarking with NVBench!
