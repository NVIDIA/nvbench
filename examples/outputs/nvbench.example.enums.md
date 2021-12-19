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
Run:  runtime_enum_sweep_string [Device=0 MyEnum=A]
Pass: Cold: 1.003842ms GPU, 1.010145ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001478ms GPU, 0.52s total GPU, 523x
Run:  runtime_enum_sweep_string [Device=0 MyEnum=B]
Pass: Cold: 1.003849ms GPU, 1.010159ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001478ms GPU, 0.52s total GPU, 523x
Run:  runtime_enum_sweep_string [Device=0 MyEnum=C]
Pass: Cold: 1.003841ms GPU, 1.010124ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001476ms GPU, 0.52s total GPU, 523x
Run:  runtime_enum_sweep_string [Device=1 MyEnum=A]
Pass: Cold: 1.002774ms GPU, 1.007653ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001473ms GPU, 0.52s total GPU, 524x
Run:  runtime_enum_sweep_string [Device=1 MyEnum=B]
Pass: Cold: 1.002610ms GPU, 1.007306ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001474ms GPU, 0.52s total GPU, 524x
Run:  runtime_enum_sweep_string [Device=1 MyEnum=C]
Pass: Cold: 1.002611ms GPU, 1.007310ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001474ms GPU, 0.52s total GPU, 524x
Run:  runtime_enum_sweep_int64 [Device=0 MyEnum=0]
Pass: Cold: 1.003846ms GPU, 1.010169ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001476ms GPU, 0.52s total GPU, 524x
Run:  runtime_enum_sweep_int64 [Device=0 MyEnum=1]
Pass: Cold: 1.003646ms GPU, 1.010107ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001476ms GPU, 0.52s total GPU, 523x
Run:  runtime_enum_sweep_int64 [Device=0 MyEnum=2]
Pass: Cold: 1.003668ms GPU, 1.010126ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001476ms GPU, 0.52s total GPU, 523x
Run:  runtime_enum_sweep_int64 [Device=1 MyEnum=0]
Pass: Cold: 1.002765ms GPU, 1.007648ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001474ms GPU, 0.52s total GPU, 524x
Run:  runtime_enum_sweep_int64 [Device=1 MyEnum=1]
Pass: Cold: 1.002585ms GPU, 1.007336ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001473ms GPU, 0.52s total GPU, 524x
Run:  runtime_enum_sweep_int64 [Device=1 MyEnum=2]
Pass: Cold: 1.002590ms GPU, 1.007300ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001475ms GPU, 0.52s total GPU, 524x
Run:  compile_time_enum_sweep [Device=0 MyEnum=A]
Pass: Cold: 1.003754ms GPU, 1.010085ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001476ms GPU, 0.52s total GPU, 523x
Run:  compile_time_enum_sweep [Device=0 MyEnum=B]
Pass: Cold: 1.003678ms GPU, 1.010054ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001474ms GPU, 0.52s total GPU, 524x
Run:  compile_time_enum_sweep [Device=0 MyEnum=C]
Pass: Cold: 1.003675ms GPU, 1.010119ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001476ms GPU, 0.52s total GPU, 524x
Run:  compile_time_enum_sweep [Device=1 MyEnum=A]
Pass: Cold: 1.002574ms GPU, 1.007283ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001473ms GPU, 0.52s total GPU, 524x
Run:  compile_time_enum_sweep [Device=1 MyEnum=B]
Pass: Cold: 1.002588ms GPU, 1.007315ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001475ms GPU, 0.52s total GPU, 524x
Run:  compile_time_enum_sweep [Device=1 MyEnum=C]
Pass: Cold: 1.002604ms GPU, 1.007322ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001475ms GPU, 0.52s total GPU, 524x
Run:  compile_time_int_sweep [Device=0 SomeInts=0]
Pass: Cold: 1.003672ms GPU, 1.010124ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001478ms GPU, 0.52s total GPU, 524x
Run:  compile_time_int_sweep [Device=0 SomeInts=16]
Pass: Cold: 1.003705ms GPU, 1.010120ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001476ms GPU, 0.52s total GPU, 523x
Run:  compile_time_int_sweep [Device=0 SomeInts=4096]
Pass: Cold: 1.003686ms GPU, 1.010048ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001476ms GPU, 0.52s total GPU, 524x
Run:  compile_time_int_sweep [Device=0 SomeInts=-12]
Pass: Cold: 1.003658ms GPU, 1.010105ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001476ms GPU, 0.52s total GPU, 523x
Run:  compile_time_int_sweep [Device=1 SomeInts=0]
Pass: Cold: 1.002591ms GPU, 1.007352ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001473ms GPU, 0.52s total GPU, 524x
Run:  compile_time_int_sweep [Device=1 SomeInts=16]
Pass: Cold: 1.002569ms GPU, 1.007280ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001474ms GPU, 0.52s total GPU, 524x
Run:  compile_time_int_sweep [Device=1 SomeInts=4096]
Pass: Cold: 1.002577ms GPU, 1.007286ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001475ms GPU, 0.52s total GPU, 524x
Run:  compile_time_int_sweep [Device=1 SomeInts=-12]
Pass: Cold: 1.002589ms GPU, 1.007329ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001473ms GPU, 0.52s total GPU, 524x
```

# Benchmark Results

## runtime_enum_sweep_string

### [0] Quadro GV100

| MyEnum | Samples | CPU Time | Noise | GPU Time | Noise | Batch GPU | Batch |
|--------|---------|----------|-------|----------|-------|-----------|-------|
|      A |    499x | 1.010 ms | 0.05% | 1.004 ms | 0.06% |  1.001 ms |  523x |
|      B |    499x | 1.010 ms | 0.05% | 1.004 ms | 0.06% |  1.001 ms |  523x |
|      C |    499x | 1.010 ms | 0.05% | 1.004 ms | 0.06% |  1.001 ms |  523x |

### [1] Quadro GP100

| MyEnum | Samples | CPU Time | Noise | GPU Time | Noise | Batch GPU | Batch |
|--------|---------|----------|-------|----------|-------|-----------|-------|
|      A |    499x | 1.008 ms | 0.05% | 1.003 ms | 0.03% |  1.001 ms |  524x |
|      B |    499x | 1.007 ms | 0.04% | 1.003 ms | 0.03% |  1.001 ms |  524x |
|      C |    499x | 1.007 ms | 0.04% | 1.003 ms | 0.03% |  1.001 ms |  524x |

## runtime_enum_sweep_int64

### [0] Quadro GV100

| MyEnum | Samples | CPU Time | Noise | GPU Time | Noise | Batch GPU | Batch |
|--------|---------|----------|-------|----------|-------|-----------|-------|
|      0 |    499x | 1.010 ms | 0.05% | 1.004 ms | 0.06% |  1.001 ms |  524x |
|      1 |    499x | 1.010 ms | 0.05% | 1.004 ms | 0.05% |  1.001 ms |  523x |
|      2 |    499x | 1.010 ms | 0.05% | 1.004 ms | 0.05% |  1.001 ms |  523x |

### [1] Quadro GP100

| MyEnum | Samples | CPU Time | Noise | GPU Time | Noise | Batch GPU | Batch |
|--------|---------|----------|-------|----------|-------|-----------|-------|
|      0 |    499x | 1.008 ms | 0.05% | 1.003 ms | 0.03% |  1.001 ms |  524x |
|      1 |    499x | 1.007 ms | 0.07% | 1.003 ms | 0.03% |  1.001 ms |  524x |
|      2 |    499x | 1.007 ms | 0.04% | 1.003 ms | 0.03% |  1.001 ms |  524x |

## compile_time_enum_sweep

### [0] Quadro GV100

| MyEnum | Samples | CPU Time | Noise | GPU Time | Noise | Batch GPU | Batch |
|--------|---------|----------|-------|----------|-------|-----------|-------|
|      A |    499x | 1.010 ms | 0.05% | 1.004 ms | 0.05% |  1.001 ms |  523x |
|      B |    499x | 1.010 ms | 0.05% | 1.004 ms | 0.06% |  1.001 ms |  524x |
|      C |    499x | 1.010 ms | 0.05% | 1.004 ms | 0.05% |  1.001 ms |  524x |

### [1] Quadro GP100

| MyEnum | Samples | CPU Time | Noise | GPU Time | Noise | Batch GPU | Batch |
|--------|---------|----------|-------|----------|-------|-----------|-------|
|      A |    499x | 1.007 ms | 0.04% | 1.003 ms | 0.03% |  1.001 ms |  524x |
|      B |    499x | 1.007 ms | 0.04% | 1.003 ms | 0.03% |  1.001 ms |  524x |
|      C |    499x | 1.007 ms | 0.04% | 1.003 ms | 0.03% |  1.001 ms |  524x |

## compile_time_int_sweep

### [0] Quadro GV100

| SomeInts | Samples | CPU Time | Noise | GPU Time | Noise | Batch GPU | Batch |
|----------|---------|----------|-------|----------|-------|-----------|-------|
|        0 |    499x | 1.010 ms | 0.05% | 1.004 ms | 0.05% |  1.001 ms |  524x |
|       16 |    499x | 1.010 ms | 0.05% | 1.004 ms | 0.06% |  1.001 ms |  523x |
|     4096 |    499x | 1.010 ms | 0.05% | 1.004 ms | 0.06% |  1.001 ms |  524x |
|      -12 |    499x | 1.010 ms | 0.05% | 1.004 ms | 0.05% |  1.001 ms |  523x |

### [1] Quadro GP100

| SomeInts | Samples | CPU Time | Noise | GPU Time | Noise | Batch GPU | Batch |
|----------|---------|----------|-------|----------|-------|-----------|-------|
|        0 |    499x | 1.007 ms | 0.05% | 1.003 ms | 0.03% |  1.001 ms |  524x |
|       16 |    499x | 1.007 ms | 0.04% | 1.003 ms | 0.03% |  1.001 ms |  524x |
|     4096 |    499x | 1.007 ms | 0.04% | 1.003 ms | 0.03% |  1.001 ms |  524x |
|      -12 |    499x | 1.007 ms | 0.04% | 1.003 ms | 0.03% |  1.001 ms |  524x |
