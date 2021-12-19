# Devices

## [0] `Quadro GV100`
* SM Version: 700 (PTX Version: 700)
* Number of SMs: 80
* SM Default Clock Rate: 1627 MHz
* Global Memory: 32163 MiB Free / 32507 MiB Total
* Global Memory Bus Peak: 870 GB/sec (4096-bit DDR @850MHz)
* Max Shared Memory: 96 KiB/SM, 48 KiB/Block
* L2 Cache Size: 6144 KiB
* Maximum Active Blocks: 32/SM
* Maximum Active Threads: 2048/SM, 1024/Block
* Available Registers: 65536/SM, 65536/Block
* ECC Enabled: No

## [1] `Quadro GP100`
* SM Version: 600 (PTX Version: 600)
* Number of SMs: 56
* SM Default Clock Rate: 1442 MHz
* Global Memory: 15999 MiB Free / 16278 MiB Total
* Global Memory Bus Peak: 732 GB/sec (4096-bit DDR @715MHz)
* Max Shared Memory: 64 KiB/SM, 48 KiB/Block
* L2 Cache Size: 4096 KiB
* Maximum Active Blocks: 32/SM
* Maximum Active Threads: 2048/SM, 1024/Block
* Available Registers: 65536/SM, 65536/Block
* ECC Enabled: No

# Log

```
Run:  throughput_bench [Device=0 T=1 Stride=1]
Pass: Cold: 0.463909ms GPU, 0.470116ms CPU, 0.50s total GPU, 1078x
Pass: Batch: 0.461110ms GPU, 0.52s total GPU, 1134x
Run:  throughput_bench [Device=0 T=1 Stride=4]
Pass: Cold: 1.106289ms GPU, 1.112558ms CPU, 0.50s total GPU, 452x
Pass: Batch: 1.102913ms GPU, 0.52s total GPU, 473x
Run:  throughput_bench [Device=0 T=2 Stride=1]
Pass: Cold: 0.913147ms GPU, 0.919417ms CPU, 0.50s total GPU, 548x
Pass: Batch: 0.910496ms GPU, 0.52s total GPU, 576x
Run:  throughput_bench [Device=0 T=2 Stride=4]
Pass: Cold: 2.868418ms GPU, 2.874656ms CPU, 0.50s total GPU, 175x
Pass: Batch: 2.855257ms GPU, 0.53s total GPU, 184x
Run:  throughput_bench [Device=1 T=1 Stride=1]
Warn: CUPTI failed to construct profiler: Device: 1 isn't supported (CC 600)
Fail: Unexpected error: Device: 1 isn't supported (CC 600)
Run:  throughput_bench [Device=1 T=1 Stride=4]
Warn: CUPTI failed to construct profiler: Device: 1 isn't supported (CC 600)
Fail: Unexpected error: Device: 1 isn't supported (CC 600)
Run:  throughput_bench [Device=1 T=2 Stride=1]
Warn: CUPTI failed to construct profiler: Device: 1 isn't supported (CC 600)
Fail: Unexpected error: Device: 1 isn't supported (CC 600)
Run:  throughput_bench [Device=1 T=2 Stride=4]
Warn: CUPTI failed to construct profiler: Device: 1 isn't supported (CC 600)
Fail: Unexpected error: Device: 1 isn't supported (CC 600)
```

# Benchmark Results

## throughput_bench

### [0] Quadro GV100

| T | Stride | Elements | HBWPeak | LoadEff | StoreEff | L1HitRate | L2HitRate | Samples |  CPU Time  | Noise |  GPU Time  | Noise | Elem/s  | Batch GPU  | Batch |
|---|--------|----------|---------|---------|----------|-----------|-----------|---------|------------|-------|------------|-------|---------|------------|-------|
| 1 |      1 | 33554432 |  65.85% | 100.00% |  100.00% |     0.00% |    50.00% |   1078x | 470.116 us | 0.14% | 463.909 us | 0.15% | 72.330G | 461.110 us | 1134x |
| 1 |      4 | 33554432 |  69.47% |  25.00% |  100.00% |     0.00% |    20.00% |    452x |   1.113 ms | 0.07% |   1.106 ms | 0.07% | 30.331G |   1.103 ms |  473x |
| 2 |      1 | 33554432 |  67.17% |  50.69% |  100.00% |    26.82% |    54.38% |    548x | 919.417 us | 0.06% | 913.147 us | 0.07% | 36.746G | 910.496 us |  576x |
| 2 |      4 | 33554432 |  58.26% |  12.67% |  100.00% |    20.69% |    38.12% |    175x |   2.875 ms | 0.25% |   2.868 ms | 0.25% | 11.698G |   2.855 ms |  184x |

### [1] Quadro GP100

No data -- check log.
