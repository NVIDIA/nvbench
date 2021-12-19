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
Run:  throughput_bench [Device=0]
Warn: Current measurement timed out (15.00s) while over noise threshold (1.15% > 0.50%)
Pass: Cold: 0.264787ms GPU, 0.270938ms CPU, 12.64s total GPU, 47755x
Pass: Batch: 0.263257ms GPU, 12.57s total GPU, 47756x
Run:  throughput_bench [Device=1]
Warn: Current measurement timed out (15.00s) while over noise threshold (1.00% > 0.50%)
Pass: Cold: 0.275773ms GPU, 0.280471ms CPU, 12.89s total GPU, 46734x
Pass: Batch: 0.275412ms GPU, 12.87s total GPU, 46735x
```

# Benchmark Results

## throughput_bench

### [0] Quadro GV100

| NumElements |  DataSize  | Samples |  CPU Time  | Noise |  GPU Time  | Noise | Elem/s  | GlobalMem BW | BWPeak | Batch GPU  | Batch  |
|-------------|------------|---------|------------|-------|------------|-------|---------|--------------|--------|------------|--------|
|    16777216 | 64.000 MiB |  47755x | 270.938 us | 1.12% | 264.787 us | 1.15% | 63.361G | 506.889 GB/s | 58.24% | 263.257 us | 47756x |

### [1] Quadro GP100

| NumElements |  DataSize  | Samples |  CPU Time  | Noise |  GPU Time  | Noise | Elem/s  | GlobalMem BW | BWPeak | Batch GPU  | Batch  |
|-------------|------------|---------|------------|-------|------------|-------|---------|--------------|--------|------------|--------|
|    16777216 | 64.000 MiB |  46734x | 280.471 us | 0.99% | 275.773 us | 1.00% | 60.837G | 486.697 GB/s | 66.47% | 275.412 us | 46735x |
