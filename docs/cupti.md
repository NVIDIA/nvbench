# CUPTI integration

NVBench can collect hardware performance metrics through CUPTI while it runs a
benchmark. CUPTI collection is optional and is enabled by default when the
CUDA Toolkit is at least 11.3 and the required CUPTI libraries are available.

## Build with CUPTI

Enable CUPTI explicitly when configuring NVBench:

```console
cmake -S . -B build \
  -DNVBench_ENABLE_CUPTI=ON \
  -DNVBench_ENABLE_EXAMPLES=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build
```

Set `NVBench_ENABLE_CUPTI=OFF` to build without CUPTI support. The default is
automatically disabled for CUDA Toolkit versions older than 11.3.

## Collect metrics in a benchmark

Include the NVBench headers and request the metrics from the benchmark state:

```cpp
state.collect_dram_throughput();
state.collect_l1_hit_rates();
state.collect_l2_hit_rates();
state.collect_loads_efficiency();
state.collect_stores_efficiency();
```

`state.collect_cupti_metrics()` enables the standard collection set shown
above. The complete example is
[`examples/auto_throughput.cu`](../examples/auto_throughput.cu).

Metric collection adds replay passes and can substantially increase runtime.
Use it for the benchmark configurations where the extra measurements are
needed. The collected values are reported with the benchmark summaries.

## Run a profiled benchmark

Build and run the example executable as usual:

```console
./build/bin/nvbench.example.auto_throughput
```

The target GPU and installed driver must support the requested CUPTI metrics.
If the device does not support CUPTI profiling, NVBench reports the profiling
error instead of silently returning metric values.

The `--profile` command-line option is separate from metric collection. It
disables NVBench instrumentation that can interfere with external profilers;
use it when profiling with an external CUDA tool.
