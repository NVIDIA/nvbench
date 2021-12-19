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
Run:  runtime_skip [Device=0 Duration=0 Kramble=Foo]
Warn: Current measurement timed out (15.00s) while over noise threshold (10.69% > 0.50%)
Pass: Cold: 0.004411ms GPU, 0.010558ms CPU, 0.65s total GPU, 148083x
Pass: Batch: 0.002086ms GPU, 0.50s total GPU, 239683x
Run:  runtime_skip [Device=0 Duration=0.00025 Kramble=Foo]
Pass: Cold: 0.254224ms GPU, 0.260565ms CPU, 0.50s total GPU, 1967x
Pass: Batch: 0.251905ms GPU, 0.52s total GPU, 2064x
Run:  runtime_skip [Device=0 Duration=0.0005 Kramble=Foo]
Skip: Long 'Foo' benchmarks are skipped.
Run:  runtime_skip [Device=0 Duration=0.00075 Kramble=Foo]
Skip: Long 'Foo' benchmarks are skipped.
Run:  runtime_skip [Device=0 Duration=0.001 Kramble=Foo]
Skip: Long 'Foo' benchmarks are skipped.
Run:  runtime_skip [Device=0 Duration=0 Kramble=Bar]
Warn: Current measurement timed out (15.00s) while over noise threshold (9.63% > 0.50%)
Pass: Cold: 0.004310ms GPU, 0.010478ms CPU, 0.64s total GPU, 147976x
Pass: Batch: 0.002103ms GPU, 0.50s total GPU, 237900x
Run:  runtime_skip [Device=0 Duration=0.00025 Kramble=Bar]
Pass: Cold: 0.254251ms GPU, 0.260605ms CPU, 0.50s total GPU, 1967x
Pass: Batch: 0.251905ms GPU, 0.52s total GPU, 2064x
Run:  runtime_skip [Device=0 Duration=0.0005 Kramble=Bar]
Pass: Cold: 0.503895ms GPU, 0.510339ms CPU, 0.50s total GPU, 993x
Pass: Batch: 0.501761ms GPU, 0.52s total GPU, 1044x
Run:  runtime_skip [Device=0 Duration=0.00075 Kramble=Bar]
Pass: Cold: 0.753776ms GPU, 0.760232ms CPU, 0.50s total GPU, 664x
Pass: Batch: 0.751619ms GPU, 0.52s total GPU, 697x
Run:  runtime_skip [Device=0 Duration=0.001 Kramble=Bar]
Pass: Cold: 1.003652ms GPU, 1.010041ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001476ms GPU, 0.52s total GPU, 523x
Run:  runtime_skip [Device=0 Duration=0 Kramble=Baz]
Skip: Short 'Baz' benchmarks are skipped.
Run:  runtime_skip [Device=0 Duration=0.00025 Kramble=Baz]
Skip: Short 'Baz' benchmarks are skipped.
Run:  runtime_skip [Device=0 Duration=0.0005 Kramble=Baz]
Skip: Short 'Baz' benchmarks are skipped.
Run:  runtime_skip [Device=0 Duration=0.00075 Kramble=Baz]
Skip: Short 'Baz' benchmarks are skipped.
Run:  runtime_skip [Device=0 Duration=0.001 Kramble=Baz]
Pass: Cold: 1.003621ms GPU, 1.010021ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001476ms GPU, 0.52s total GPU, 524x
Run:  runtime_skip [Device=1 Duration=0 Kramble=Foo]
Warn: Current measurement timed out (15.00s) while over noise threshold (4.20% > 0.50%)
Warn: Current measurement timed out (15.00s) before accumulating min_time (0.47s < 0.50s)
Pass: Cold: 0.003054ms GPU, 0.007790ms CPU, 0.47s total GPU, 152833x
Pass: Batch: 0.001348ms GPU, 0.50s total GPU, 371096x
Run:  runtime_skip [Device=1 Duration=0.00025 Kramble=Foo]
Pass: Cold: 0.253030ms GPU, 0.257734ms CPU, 0.50s total GPU, 1977x
Pass: Batch: 0.251905ms GPU, 0.52s total GPU, 2073x
Run:  runtime_skip [Device=1 Duration=0.0005 Kramble=Foo]
Skip: Long 'Foo' benchmarks are skipped.
Run:  runtime_skip [Device=1 Duration=0.00075 Kramble=Foo]
Skip: Long 'Foo' benchmarks are skipped.
Run:  runtime_skip [Device=1 Duration=0.001 Kramble=Foo]
Skip: Long 'Foo' benchmarks are skipped.
Run:  runtime_skip [Device=1 Duration=0 Kramble=Bar]
Warn: Current measurement timed out (15.00s) while over noise threshold (6.80% > 0.50%)
Warn: Current measurement timed out (15.00s) before accumulating min_time (0.48s < 0.50s)
Pass: Cold: 0.003132ms GPU, 0.007898ms CPU, 0.48s total GPU, 152569x
Pass: Batch: 0.001441ms GPU, 0.50s total GPU, 346971x
Run:  runtime_skip [Device=1 Duration=0.00025 Kramble=Bar]
Pass: Cold: 0.253034ms GPU, 0.257721ms CPU, 0.50s total GPU, 1977x
Pass: Batch: 0.251905ms GPU, 0.52s total GPU, 2074x
Run:  runtime_skip [Device=1 Duration=0.0005 Kramble=Bar]
Pass: Cold: 0.502882ms GPU, 0.507600ms CPU, 0.50s total GPU, 995x
Pass: Batch: 0.501761ms GPU, 0.52s total GPU, 1045x
Run:  runtime_skip [Device=1 Duration=0.00075 Kramble=Bar]
Pass: Cold: 0.752730ms GPU, 0.757417ms CPU, 0.50s total GPU, 665x
Pass: Batch: 0.751617ms GPU, 0.52s total GPU, 698x
Run:  runtime_skip [Device=1 Duration=0.001 Kramble=Bar]
Pass: Cold: 1.002601ms GPU, 1.007330ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001474ms GPU, 0.52s total GPU, 524x
Run:  runtime_skip [Device=1 Duration=0 Kramble=Baz]
Skip: Short 'Baz' benchmarks are skipped.
Run:  runtime_skip [Device=1 Duration=0.00025 Kramble=Baz]
Skip: Short 'Baz' benchmarks are skipped.
Run:  runtime_skip [Device=1 Duration=0.0005 Kramble=Baz]
Skip: Short 'Baz' benchmarks are skipped.
Run:  runtime_skip [Device=1 Duration=0.00075 Kramble=Baz]
Skip: Short 'Baz' benchmarks are skipped.
Run:  runtime_skip [Device=1 Duration=0.001 Kramble=Baz]
Pass: Cold: 1.002592ms GPU, 1.007267ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001475ms GPU, 0.52s total GPU, 524x
Run:  skip_overload [Device=0 In=I32 Out=I32]
Skip: InputType == OutputType.
Run:  skip_overload [Device=0 In=I32 Out=I64]
Pass: Cold: 1.003637ms GPU, 1.010116ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001478ms GPU, 0.52s total GPU, 523x
Run:  skip_overload [Device=0 In=I64 Out=I32]
Pass: Cold: 1.003626ms GPU, 1.010094ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001476ms GPU, 0.52s total GPU, 523x
Run:  skip_overload [Device=0 In=I64 Out=I64]
Skip: InputType == OutputType.
Run:  skip_overload [Device=1 In=I32 Out=I32]
Skip: InputType == OutputType.
Run:  skip_overload [Device=1 In=I32 Out=I64]
Pass: Cold: 1.002573ms GPU, 1.007294ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001473ms GPU, 0.52s total GPU, 524x
Run:  skip_overload [Device=1 In=I64 Out=I32]
Pass: Cold: 1.002588ms GPU, 1.007258ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001474ms GPU, 0.52s total GPU, 524x
Run:  skip_overload [Device=1 In=I64 Out=I64]
Skip: InputType == OutputType.
Run:  skip_sfinae [Device=0 In=I8 Out=I8]
Pass: Cold: 1.003653ms GPU, 1.010138ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001476ms GPU, 0.52s total GPU, 524x
Run:  skip_sfinae [Device=0 In=I8 Out=I16]
Pass: Cold: 1.003682ms GPU, 1.010137ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001476ms GPU, 0.52s total GPU, 524x
Run:  skip_sfinae [Device=0 In=I8 Out=I32]
Pass: Cold: 1.003715ms GPU, 1.010145ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001478ms GPU, 0.52s total GPU, 523x
Run:  skip_sfinae [Device=0 In=I8 Out=I64]
Pass: Cold: 1.003687ms GPU, 1.010252ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001478ms GPU, 0.52s total GPU, 524x
Run:  skip_sfinae [Device=0 In=I16 Out=I8]
Skip: sizeof(InputType) > sizeof(OutputType).
Run:  skip_sfinae [Device=0 In=I16 Out=I16]
Pass: Cold: 1.003672ms GPU, 1.010219ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001476ms GPU, 0.52s total GPU, 523x
Run:  skip_sfinae [Device=0 In=I16 Out=I32]
Pass: Cold: 1.003689ms GPU, 1.010125ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001478ms GPU, 0.52s total GPU, 524x
Run:  skip_sfinae [Device=0 In=I16 Out=I64]
Pass: Cold: 1.003709ms GPU, 1.010160ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001476ms GPU, 0.52s total GPU, 523x
Run:  skip_sfinae [Device=0 In=I32 Out=I8]
Skip: sizeof(InputType) > sizeof(OutputType).
Run:  skip_sfinae [Device=0 In=I32 Out=I16]
Skip: sizeof(InputType) > sizeof(OutputType).
Run:  skip_sfinae [Device=0 In=I32 Out=I32]
Pass: Cold: 1.003687ms GPU, 1.010157ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001476ms GPU, 0.52s total GPU, 523x
Run:  skip_sfinae [Device=0 In=I32 Out=I64]
Pass: Cold: 1.003693ms GPU, 1.010184ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001476ms GPU, 0.52s total GPU, 524x
Run:  skip_sfinae [Device=0 In=I64 Out=I8]
Skip: sizeof(InputType) > sizeof(OutputType).
Run:  skip_sfinae [Device=0 In=I64 Out=I16]
Skip: sizeof(InputType) > sizeof(OutputType).
Run:  skip_sfinae [Device=0 In=I64 Out=I32]
Skip: sizeof(InputType) > sizeof(OutputType).
Run:  skip_sfinae [Device=0 In=I64 Out=I64]
Pass: Cold: 1.003664ms GPU, 1.010159ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001478ms GPU, 0.52s total GPU, 523x
Run:  skip_sfinae [Device=1 In=I8 Out=I8]
Pass: Cold: 1.002595ms GPU, 1.007262ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001474ms GPU, 0.52s total GPU, 524x
Run:  skip_sfinae [Device=1 In=I8 Out=I16]
Pass: Cold: 1.002599ms GPU, 1.007262ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001475ms GPU, 0.52s total GPU, 524x
Run:  skip_sfinae [Device=1 In=I8 Out=I32]
Pass: Cold: 1.002558ms GPU, 1.007394ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001475ms GPU, 0.52s total GPU, 524x
Run:  skip_sfinae [Device=1 In=I8 Out=I64]
Pass: Cold: 1.002551ms GPU, 1.007314ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001475ms GPU, 0.52s total GPU, 524x
Run:  skip_sfinae [Device=1 In=I16 Out=I8]
Skip: sizeof(InputType) > sizeof(OutputType).
Run:  skip_sfinae [Device=1 In=I16 Out=I16]
Pass: Cold: 1.002570ms GPU, 1.007255ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001474ms GPU, 0.52s total GPU, 524x
Run:  skip_sfinae [Device=1 In=I16 Out=I32]
Pass: Cold: 1.002602ms GPU, 1.007275ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001474ms GPU, 0.52s total GPU, 524x
Run:  skip_sfinae [Device=1 In=I16 Out=I64]
Pass: Cold: 1.002582ms GPU, 1.007273ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001474ms GPU, 0.52s total GPU, 524x
Run:  skip_sfinae [Device=1 In=I32 Out=I8]
Skip: sizeof(InputType) > sizeof(OutputType).
Run:  skip_sfinae [Device=1 In=I32 Out=I16]
Skip: sizeof(InputType) > sizeof(OutputType).
Run:  skip_sfinae [Device=1 In=I32 Out=I32]
Pass: Cold: 1.002593ms GPU, 1.007290ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001474ms GPU, 0.52s total GPU, 524x
Run:  skip_sfinae [Device=1 In=I32 Out=I64]
Pass: Cold: 1.002575ms GPU, 1.007244ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001475ms GPU, 0.52s total GPU, 524x
Run:  skip_sfinae [Device=1 In=I64 Out=I8]
Skip: sizeof(InputType) > sizeof(OutputType).
Run:  skip_sfinae [Device=1 In=I64 Out=I16]
Skip: sizeof(InputType) > sizeof(OutputType).
Run:  skip_sfinae [Device=1 In=I64 Out=I32]
Skip: sizeof(InputType) > sizeof(OutputType).
Run:  skip_sfinae [Device=1 In=I64 Out=I64]
Pass: Cold: 1.002587ms GPU, 1.007259ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001475ms GPU, 0.52s total GPU, 524x
```

# Benchmark Results

## runtime_skip

### [0] Quadro GV100

| Duration | Kramble | Samples |  CPU Time  | Noise |  GPU Time  | Noise  | Batch GPU  |  Batch  |
|----------|---------|---------|------------|-------|------------|--------|------------|---------|
|        0 |     Foo | 148083x |  10.558 us | 3.46% |   4.411 us | 10.69% |   2.086 us | 239683x |
|  0.00025 |     Foo |   1967x | 260.565 us | 0.18% | 254.224 us |  0.23% | 251.905 us |   2064x |
|        0 |     Bar | 147976x |  10.478 us | 5.19% |   4.310 us |  9.63% |   2.103 us | 237900x |
|  0.00025 |     Bar |   1967x | 260.605 us | 0.44% | 254.251 us |  0.19% | 251.905 us |   2064x |
|   0.0005 |     Bar |    993x | 510.339 us | 0.09% | 503.895 us |  0.09% | 501.761 us |   1044x |
|  0.00075 |     Bar |    664x | 760.232 us | 0.12% | 753.776 us |  0.07% | 751.619 us |    697x |
|    0.001 |     Bar |    499x |   1.010 ms | 0.05% |   1.004 ms |  0.05% |   1.001 ms |    523x |
|    0.001 |     Baz |    499x |   1.010 ms | 0.05% |   1.004 ms |  0.05% |   1.001 ms |    524x |

### [1] Quadro GP100

| Duration | Kramble | Samples |  CPU Time  | Noise |  GPU Time  | Noise | Batch GPU  |  Batch  |
|----------|---------|---------|------------|-------|------------|-------|------------|---------|
|        0 |     Foo | 152833x |   7.790 us | 5.25% |   3.054 us | 4.20% |   1.348 us | 371096x |
|  0.00025 |     Foo |   1977x | 257.734 us | 0.17% | 253.030 us | 0.12% | 251.905 us |   2073x |
|        0 |     Bar | 152569x |   7.898 us | 6.09% |   3.132 us | 6.80% |   1.441 us | 346971x |
|  0.00025 |     Bar |   1977x | 257.721 us | 0.27% | 253.034 us | 0.13% | 251.905 us |   2074x |
|   0.0005 |     Bar |    995x | 507.600 us | 0.08% | 502.882 us | 0.06% | 501.761 us |   1045x |
|  0.00075 |     Bar |    665x | 757.417 us | 0.05% | 752.730 us | 0.04% | 751.617 us |    698x |
|    0.001 |     Bar |    499x |   1.007 ms | 0.04% |   1.003 ms | 0.03% |   1.001 ms |    524x |
|    0.001 |     Baz |    499x |   1.007 ms | 0.04% |   1.003 ms | 0.03% |   1.001 ms |    524x |

## skip_overload

### [0] Quadro GV100

| In  | Out | Samples | CPU Time | Noise | GPU Time | Noise | Batch GPU | Batch |
|-----|-----|---------|----------|-------|----------|-------|-----------|-------|
| I32 | I64 |    499x | 1.010 ms | 0.05% | 1.004 ms | 0.05% |  1.001 ms |  523x |
| I64 | I32 |    499x | 1.010 ms | 0.05% | 1.004 ms | 0.05% |  1.001 ms |  523x |

### [1] Quadro GP100

| In  | Out | Samples | CPU Time | Noise | GPU Time | Noise | Batch GPU | Batch |
|-----|-----|---------|----------|-------|----------|-------|-----------|-------|
| I32 | I64 |    499x | 1.007 ms | 0.04% | 1.003 ms | 0.03% |  1.001 ms |  524x |
| I64 | I32 |    499x | 1.007 ms | 0.04% | 1.003 ms | 0.03% |  1.001 ms |  524x |

## skip_sfinae

### [0] Quadro GV100

| In  | Out | Samples | CPU Time | Noise | GPU Time | Noise | Batch GPU | Batch |
|-----|-----|---------|----------|-------|----------|-------|-----------|-------|
|  I8 |  I8 |    499x | 1.010 ms | 0.05% | 1.004 ms | 0.05% |  1.001 ms |  524x |
|  I8 | I16 |    499x | 1.010 ms | 0.05% | 1.004 ms | 0.05% |  1.001 ms |  524x |
|  I8 | I32 |    499x | 1.010 ms | 0.05% | 1.004 ms | 0.05% |  1.001 ms |  523x |
|  I8 | I64 |    499x | 1.010 ms | 0.05% | 1.004 ms | 0.05% |  1.001 ms |  524x |
| I16 | I16 |    499x | 1.010 ms | 0.05% | 1.004 ms | 0.05% |  1.001 ms |  523x |
| I16 | I32 |    499x | 1.010 ms | 0.05% | 1.004 ms | 0.05% |  1.001 ms |  524x |
| I16 | I64 |    499x | 1.010 ms | 0.04% | 1.004 ms | 0.05% |  1.001 ms |  523x |
| I32 | I32 |    499x | 1.010 ms | 0.05% | 1.004 ms | 0.05% |  1.001 ms |  523x |
| I32 | I64 |    499x | 1.010 ms | 0.05% | 1.004 ms | 0.05% |  1.001 ms |  524x |
| I64 | I64 |    499x | 1.010 ms | 0.04% | 1.004 ms | 0.05% |  1.001 ms |  523x |

### [1] Quadro GP100

| In  | Out | Samples | CPU Time | Noise | GPU Time | Noise | Batch GPU | Batch |
|-----|-----|---------|----------|-------|----------|-------|-----------|-------|
|  I8 |  I8 |    499x | 1.007 ms | 0.04% | 1.003 ms | 0.03% |  1.001 ms |  524x |
|  I8 | I16 |    499x | 1.007 ms | 0.04% | 1.003 ms | 0.03% |  1.001 ms |  524x |
|  I8 | I32 |    499x | 1.007 ms | 0.05% | 1.003 ms | 0.03% |  1.001 ms |  524x |
|  I8 | I64 |    499x | 1.007 ms | 0.04% | 1.003 ms | 0.03% |  1.001 ms |  524x |
| I16 | I16 |    499x | 1.007 ms | 0.04% | 1.003 ms | 0.03% |  1.001 ms |  524x |
| I16 | I32 |    499x | 1.007 ms | 0.04% | 1.003 ms | 0.03% |  1.001 ms |  524x |
| I16 | I64 |    499x | 1.007 ms | 0.04% | 1.003 ms | 0.03% |  1.001 ms |  524x |
| I32 | I32 |    499x | 1.007 ms | 0.04% | 1.003 ms | 0.03% |  1.001 ms |  524x |
| I32 | I64 |    499x | 1.007 ms | 0.04% | 1.003 ms | 0.03% |  1.001 ms |  524x |
| I64 | I64 |    499x | 1.007 ms | 0.04% | 1.003 ms | 0.03% |  1.001 ms |  524x |
