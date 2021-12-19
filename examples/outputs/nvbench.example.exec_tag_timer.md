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
Run:  mod2_inplace [Device=0]
Warn: Current measurement timed out (15.00s) while over noise threshold (0.98% > 0.50%)
Pass: Cold: 0.263622ms GPU, 0.269797ms CPU, 7.27s total GPU, 27572x
Run:  mod2_inplace [Device=1]
Warn: Current measurement timed out (15.00s) while over noise threshold (0.58% > 0.50%)
Pass: Cold: 0.268388ms GPU, 0.273104ms CPU, 7.17s total GPU, 26721x
```

# Benchmark Results

## mod2_inplace

### [0] Quadro GV100

| Samples |  CPU Time  | Noise |  GPU Time  | Noise | Elem/s  | GlobalMem BW | BWPeak |
|---------|------------|-------|------------|-------|---------|--------------|--------|
|  27572x | 269.797 us | 0.96% | 263.622 us | 0.98% | 63.641G | 509.129 GB/s | 58.49% |

### [1] Quadro GP100

| Samples |  CPU Time  | Noise |  GPU Time  | Noise | Elem/s  | GlobalMem BW | BWPeak |
|---------|------------|-------|------------|-------|---------|--------------|--------|
|  26721x | 273.104 us | 0.58% | 268.388 us | 0.58% | 62.511G | 500.088 GB/s | 68.30% |
