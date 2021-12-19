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
Run:  sequence_bench [Device=0]
Warn: Current measurement timed out (15.00s) while over noise threshold (0.52% > 0.50%)
Pass: Cold: 0.107383ms GPU, 0.112108ms CPU, 9.46s total GPU, 88096x
Run:  sequence_bench [Device=1]
Pass: Cold: 0.118053ms GPU, 0.122040ms CPU, 0.50s total GPU, 4236x
```

# Benchmark Results

## sequence_bench

### [0] Quadro GV100

|  Items   |    Size    | Samples |  CPU Time  | Noise |  GPU Time  | Noise |  Elem/s  | GlobalMem BW | BWPeak |
|----------|------------|---------|------------|-------|------------|-------|----------|--------------|--------|
| 16777216 | 64.000 MiB |  88096x | 112.108 us | 0.44% | 107.383 us | 0.52% | 156.238G | 624.951 GB/s | 71.80% |

### [1] Quadro GP100

|  Items   |    Size    | Samples |  CPU Time  | Noise |  GPU Time  | Noise |  Elem/s  | GlobalMem BW | BWPeak |
|----------|------------|---------|------------|-------|------------|-------|----------|--------------|--------|
| 16777216 | 64.000 MiB |   4236x | 122.040 us | 0.32% | 118.053 us | 0.31% | 142.116G | 568.464 GB/s | 77.64% |
