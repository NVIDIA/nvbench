# Devices

## [0] `Quadro GV100`
* SM Version: 700 (PTX Version: 700)
* Number of SMs: 80
* SM Default Clock Rate: 1627 MHz
* Global Memory: 29776 MiB Free / 32507 MiB Total
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
* Global Memory: 14335 MiB Free / 16278 MiB Total
* Global Memory Bus Peak: 732 GB/sec (4096-bit DDR @715MHz)
* Max Shared Memory: 64 KiB/SM, 48 KiB/Block
* L2 Cache Size: 4096 KiB
* Maximum Active Blocks: 32/SM
* Maximum Active Threads: 2048/SM, 1024/Block
* Available Registers: 65536/SM, 65536/Block
* ECC Enabled: No

# Log

```
Run:  simple [Device=0]
Pass: Cold: 1.003764ms GPU, 1.010252ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001476ms GPU, 0.52s total GPU, 524x
Run:  simple [Device=1]
Pass: Cold: 1.002567ms GPU, 1.007237ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001475ms GPU, 0.52s total GPU, 524x
Run:  single_float64_axis [Device=0 Duration=0]
Warn: Current measurement timed out (15.00s) while over noise threshold (10.78% > 0.50%)
Pass: Cold: 0.004424ms GPU, 0.010618ms CPU, 0.65s total GPU, 147957x
Pass: Batch: 0.002043ms GPU, 0.50s total GPU, 244766x
Run:  single_float64_axis [Device=0 Duration=0.0001]
Pass: Cold: 0.103515ms GPU, 0.110048ms CPU, 0.50s total GPU, 4831x
Pass: Batch: 0.101376ms GPU, 0.52s total GPU, 5088x
Run:  single_float64_axis [Device=0 Duration=0.0002]
Pass: Cold: 0.203903ms GPU, 0.210369ms CPU, 0.50s total GPU, 2453x
Pass: Batch: 0.201729ms GPU, 0.52s total GPU, 2582x
Run:  single_float64_axis [Device=0 Duration=0.0003]
Pass: Cold: 0.303412ms GPU, 0.309866ms CPU, 0.50s total GPU, 1648x
Pass: Batch: 0.301164ms GPU, 0.52s total GPU, 1736x
Run:  single_float64_axis [Device=0 Duration=0.0004]
Pass: Cold: 0.403673ms GPU, 0.410148ms CPU, 0.50s total GPU, 1239x
Pass: Batch: 0.401410ms GPU, 0.52s total GPU, 1304x
Run:  single_float64_axis [Device=0 Duration=0.0005]
Pass: Cold: 0.504089ms GPU, 0.510529ms CPU, 0.50s total GPU, 992x
Pass: Batch: 0.501762ms GPU, 0.52s total GPU, 1042x
Run:  single_float64_axis [Device=0 Duration=0.0006]
Pass: Cold: 0.603471ms GPU, 0.609862ms CPU, 0.50s total GPU, 829x
Pass: Batch: 0.601104ms GPU, 0.52s total GPU, 872x
Run:  single_float64_axis [Device=0 Duration=0.0007]
Pass: Cold: 0.703744ms GPU, 0.710294ms CPU, 0.50s total GPU, 711x
Pass: Batch: 0.701443ms GPU, 0.52s total GPU, 748x
Run:  single_float64_axis [Device=0 Duration=0.0008]
Pass: Cold: 0.804187ms GPU, 0.810565ms CPU, 0.50s total GPU, 622x
Pass: Batch: 0.801795ms GPU, 0.52s total GPU, 653x
Run:  single_float64_axis [Device=0 Duration=0.0009]
Pass: Cold: 0.903433ms GPU, 0.909873ms CPU, 0.50s total GPU, 554x
Pass: Batch: 0.901125ms GPU, 0.52s total GPU, 582x
Run:  single_float64_axis [Device=0 Duration=0.001]
Pass: Cold: 1.003807ms GPU, 1.010270ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001476ms GPU, 0.52s total GPU, 523x
Run:  single_float64_axis [Device=1 Duration=0]
Warn: Current measurement timed out (15.00s) while over noise threshold (4.13% > 0.50%)
Warn: Current measurement timed out (15.00s) before accumulating min_time (0.46s < 0.50s)
Pass: Cold: 0.003016ms GPU, 0.007705ms CPU, 0.46s total GPU, 152839x
Pass: Batch: 0.001343ms GPU, 0.50s total GPU, 372166x
Run:  single_float64_axis [Device=1 Duration=0.0001]
Pass: Cold: 0.102481ms GPU, 0.107156ms CPU, 0.50s total GPU, 4879x
Pass: Batch: 0.101376ms GPU, 0.52s total GPU, 5107x
Run:  single_float64_axis [Device=1 Duration=0.0002]
Pass: Cold: 0.202833ms GPU, 0.207544ms CPU, 0.50s total GPU, 2466x
Pass: Batch: 0.201728ms GPU, 0.52s total GPU, 2586x
Run:  single_float64_axis [Device=1 Duration=0.0003]
Pass: Cold: 0.302191ms GPU, 0.306880ms CPU, 0.50s total GPU, 1655x
Pass: Batch: 0.301057ms GPU, 0.52s total GPU, 1736x
Run:  single_float64_axis [Device=1 Duration=0.0004]
Pass: Cold: 0.402508ms GPU, 0.407214ms CPU, 0.50s total GPU, 1243x
Pass: Batch: 0.401409ms GPU, 0.52s total GPU, 1305x
Run:  single_float64_axis [Device=1 Duration=0.0005]
Pass: Cold: 0.502864ms GPU, 0.507562ms CPU, 0.50s total GPU, 995x
Pass: Batch: 0.501761ms GPU, 0.52s total GPU, 1045x
Run:  single_float64_axis [Device=1 Duration=0.0006]
Pass: Cold: 0.602223ms GPU, 0.606954ms CPU, 0.50s total GPU, 831x
Pass: Batch: 0.601089ms GPU, 0.52s total GPU, 873x
Run:  single_float64_axis [Device=1 Duration=0.0007]
Pass: Cold: 0.702559ms GPU, 0.707255ms CPU, 0.50s total GPU, 712x
Pass: Batch: 0.701442ms GPU, 0.52s total GPU, 748x
Run:  single_float64_axis [Device=1 Duration=0.0008]
Pass: Cold: 0.802910ms GPU, 0.807636ms CPU, 0.50s total GPU, 623x
Pass: Batch: 0.801794ms GPU, 0.53s total GPU, 655x
Run:  single_float64_axis [Device=1 Duration=0.0009]
Pass: Cold: 0.902248ms GPU, 0.906935ms CPU, 0.50s total GPU, 555x
Pass: Batch: 0.901123ms GPU, 0.52s total GPU, 582x
Run:  single_float64_axis [Device=1 Duration=0.001]
Pass: Cold: 1.002594ms GPU, 1.007296ms CPU, 0.50s total GPU, 499x
Pass: Batch: 1.001473ms GPU, 0.52s total GPU, 524x
Run:  copy_sweep_grid_shape [Device=0 BlockSize=2^6 NumBlocks=2^6]
Pass: Cold: 7.615783ms GPU, 7.622137ms CPU, 0.50s total GPU, 66x
Pass: Batch: 7.614613ms GPU, 0.53s total GPU, 69x
Run:  copy_sweep_grid_shape [Device=0 BlockSize=2^8 NumBlocks=2^6]
Pass: Cold: 2.436121ms GPU, 2.442463ms CPU, 0.50s total GPU, 206x
Pass: Batch: 2.433948ms GPU, 0.52s total GPU, 215x
Run:  copy_sweep_grid_shape [Device=0 BlockSize=2^10 NumBlocks=2^6]
Warn: Current measurement timed out (15.00s) while over noise threshold (1.30% > 0.50%)
Pass: Cold: 1.105935ms GPU, 1.112225ms CPU, 14.56s total GPU, 13161x
Pass: Batch: 1.102809ms GPU, 14.52s total GPU, 13162x
Run:  copy_sweep_grid_shape [Device=0 BlockSize=2^6 NumBlocks=2^8]
Pass: Cold: 2.444184ms GPU, 2.450441ms CPU, 0.92s total GPU, 375x
Pass: Batch: 2.444397ms GPU, 0.92s total GPU, 376x
Run:  copy_sweep_grid_shape [Device=0 BlockSize=2^8 NumBlocks=2^8]
Warn: Current measurement timed out (15.00s) while over noise threshold (0.97% > 0.50%)
Pass: Cold: 1.076641ms GPU, 1.082913ms CPU, 14.54s total GPU, 13509x
Pass: Batch: 1.075515ms GPU, 14.53s total GPU, 13510x
Run:  copy_sweep_grid_shape [Device=0 BlockSize=2^10 NumBlocks=2^8]
Warn: Current measurement timed out (15.00s) while over noise threshold (0.51% > 0.50%)
Pass: Cold: 0.958478ms GPU, 0.964751ms CPU, 14.48s total GPU, 15105x
Pass: Batch: 0.957249ms GPU, 14.46s total GPU, 15106x
Run:  copy_sweep_grid_shape [Device=0 BlockSize=2^6 NumBlocks=2^10]
Warn: Current measurement timed out (15.00s) while over noise threshold (0.74% > 0.50%)
Pass: Cold: 1.070616ms GPU, 1.076881ms CPU, 14.54s total GPU, 13582x
Pass: Batch: 1.070915ms GPU, 14.55s total GPU, 13583x
Run:  copy_sweep_grid_shape [Device=0 BlockSize=2^8 NumBlocks=2^10]
Pass: Cold: 0.956568ms GPU, 0.962899ms CPU, 1.70s total GPU, 1782x
Pass: Batch: 0.954599ms GPU, 1.70s total GPU, 1783x
Run:  copy_sweep_grid_shape [Device=0 BlockSize=2^10 NumBlocks=2^10]
Warn: Current measurement timed out (15.00s) while over noise threshold (1.94% > 0.50%)
Pass: Cold: 0.994218ms GPU, 1.000471ms CPU, 14.49s total GPU, 14579x
Pass: Batch: 0.992819ms GPU, 14.48s total GPU, 14580x
Run:  copy_sweep_grid_shape [Device=1 BlockSize=2^6 NumBlocks=2^6]
Warn: Current measurement timed out (15.00s) while over noise threshold (1.10% > 0.50%)
Pass: Cold: 6.684226ms GPU, 6.688950ms CPU, 14.95s total GPU, 2236x
Pass: Batch: 6.674803ms GPU, 14.93s total GPU, 2237x
Run:  copy_sweep_grid_shape [Device=1 BlockSize=2^8 NumBlocks=2^6]
Pass: Cold: 2.296344ms GPU, 2.301080ms CPU, 0.50s total GPU, 218x
Pass: Batch: 2.298271ms GPU, 0.52s total GPU, 228x
Run:  copy_sweep_grid_shape [Device=1 BlockSize=2^10 NumBlocks=2^6]
Pass: Cold: 1.174374ms GPU, 1.179136ms CPU, 0.50s total GPU, 426x
Pass: Batch: 1.172158ms GPU, 0.53s total GPU, 449x
Run:  copy_sweep_grid_shape [Device=1 BlockSize=2^6 NumBlocks=2^8]
Pass: Cold: 2.213621ms GPU, 2.218381ms CPU, 0.50s total GPU, 226x
Pass: Batch: 2.213030ms GPU, 0.52s total GPU, 237x
Run:  copy_sweep_grid_shape [Device=1 BlockSize=2^8 NumBlocks=2^8]
Warn: Current measurement timed out (15.00s) while over noise threshold (0.67% > 0.50%)
Pass: Cold: 1.130514ms GPU, 1.135236ms CPU, 14.62s total GPU, 12933x
Pass: Batch: 1.130124ms GPU, 14.62s total GPU, 12934x
Run:  copy_sweep_grid_shape [Device=1 BlockSize=2^10 NumBlocks=2^8]
Pass: Cold: 1.118955ms GPU, 1.123642ms CPU, 0.50s total GPU, 447x
Pass: Batch: 1.117003ms GPU, 0.52s total GPU, 468x
Run:  copy_sweep_grid_shape [Device=1 BlockSize=2^6 NumBlocks=2^10]
Pass: Cold: 1.116924ms GPU, 1.121675ms CPU, 0.50s total GPU, 448x
Pass: Batch: 1.114889ms GPU, 0.52s total GPU, 470x
Run:  copy_sweep_grid_shape [Device=1 BlockSize=2^8 NumBlocks=2^10]
Pass: Cold: 1.117701ms GPU, 1.122388ms CPU, 0.50s total GPU, 448x
Pass: Batch: 1.115605ms GPU, 0.53s total GPU, 471x
Run:  copy_sweep_grid_shape [Device=1 BlockSize=2^10 NumBlocks=2^10]
Pass: Cold: 1.055706ms GPU, 1.060387ms CPU, 0.50s total GPU, 474x
Pass: Batch: 1.054097ms GPU, 0.52s total GPU, 498x
Run:  copy_type_sweep [Device=0 T=U8]
Pass: Cold: 2.543548ms GPU, 2.549831ms CPU, 0.50s total GPU, 197x
Pass: Batch: 2.539371ms GPU, 0.52s total GPU, 206x
Run:  copy_type_sweep [Device=0 T=U16]
Pass: Cold: 1.595621ms GPU, 1.601868ms CPU, 0.50s total GPU, 314x
Pass: Batch: 1.591500ms GPU, 0.53s total GPU, 331x
Run:  copy_type_sweep [Device=0 T=U32]
Warn: Current measurement timed out (15.00s) while over noise threshold (0.97% > 0.50%)
Pass: Cold: 1.076611ms GPU, 1.082876ms CPU, 14.54s total GPU, 13509x
Pass: Batch: 1.075475ms GPU, 14.53s total GPU, 13510x
Run:  copy_type_sweep [Device=0 T=U64]
Warn: Current measurement timed out (15.00s) while over noise threshold (0.61% > 0.50%)
Pass: Cold: 0.930416ms GPU, 0.936680ms CPU, 14.46s total GPU, 15542x
Pass: Batch: 0.929189ms GPU, 14.44s total GPU, 15543x
Run:  copy_type_sweep [Device=0 T=F32]
Warn: Current measurement timed out (15.00s) while over noise threshold (0.97% > 0.50%)
Pass: Cold: 1.076786ms GPU, 1.083044ms CPU, 14.55s total GPU, 13508x
Pass: Batch: 1.075385ms GPU, 14.53s total GPU, 13509x
Run:  copy_type_sweep [Device=0 T=F64]
Warn: Current measurement timed out (15.00s) while over noise threshold (0.60% > 0.50%)
Pass: Cold: 0.930412ms GPU, 0.936683ms CPU, 14.46s total GPU, 15546x
Pass: Batch: 0.929182ms GPU, 14.45s total GPU, 15547x
Run:  copy_type_sweep [Device=1 T=U8]
Warn: Current measurement timed out (15.00s) while over noise threshold (0.64% > 0.50%)
Pass: Cold: 2.703092ms GPU, 2.707829ms CPU, 14.86s total GPU, 5497x
Pass: Batch: 2.699732ms GPU, 14.84s total GPU, 5498x
Run:  copy_type_sweep [Device=1 T=U16]
Pass: Cold: 1.515335ms GPU, 1.520048ms CPU, 0.50s total GPU, 330x
Pass: Batch: 1.513689ms GPU, 0.53s total GPU, 348x
Run:  copy_type_sweep [Device=1 T=U32]
Warn: Current measurement timed out (15.00s) while over noise threshold (0.66% > 0.50%)
Pass: Cold: 1.130622ms GPU, 1.135307ms CPU, 14.62s total GPU, 12935x
Pass: Batch: 1.130123ms GPU, 14.62s total GPU, 12936x
Run:  copy_type_sweep [Device=1 T=U64]
Pass: Cold: 1.047513ms GPU, 1.052201ms CPU, 0.50s total GPU, 478x
Pass: Batch: 1.044906ms GPU, 0.52s total GPU, 500x
Run:  copy_type_sweep [Device=1 T=F32]
Warn: Current measurement timed out (15.00s) while over noise threshold (0.67% > 0.50%)
Pass: Cold: 1.130631ms GPU, 1.135341ms CPU, 14.62s total GPU, 12933x
Pass: Batch: 1.130335ms GPU, 14.62s total GPU, 12934x
Run:  copy_type_sweep [Device=1 T=F64]
Pass: Cold: 1.048417ms GPU, 1.053125ms CPU, 0.50s total GPU, 477x
Pass: Batch: 1.045540ms GPU, 0.52s total GPU, 497x
Run:  copy_type_conversion_sweep [Device=0 In=I8 Out=I8]
Skip: Not a conversion: InputType == OutputType.
Run:  copy_type_conversion_sweep [Device=0 In=I8 Out=I16]
Pass: Cold: 0.702933ms GPU, 0.709177ms CPU, 0.50s total GPU, 712x
Pass: Batch: 0.699714ms GPU, 0.52s total GPU, 748x
Run:  copy_type_conversion_sweep [Device=0 In=I8 Out=I32]
Pass: Cold: 0.804698ms GPU, 0.810934ms CPU, 0.50s total GPU, 622x
Pass: Batch: 0.802089ms GPU, 0.52s total GPU, 654x
Run:  copy_type_conversion_sweep [Device=0 In=I8 Out=F32]
Pass: Cold: 0.814768ms GPU, 0.821028ms CPU, 0.50s total GPU, 614x
Pass: Batch: 0.812088ms GPU, 0.52s total GPU, 645x
Run:  copy_type_conversion_sweep [Device=0 In=I8 Out=I64]
Warn: Current measurement timed out (15.00s) while over noise threshold (1.04% > 0.50%)
Pass: Cold: 1.212034ms GPU, 1.218286ms CPU, 14.60s total GPU, 12047x
Pass: Batch: 1.210944ms GPU, 14.59s total GPU, 12048x
Run:  copy_type_conversion_sweep [Device=0 In=I8 Out=F64]
Warn: Current measurement timed out (15.00s) while over noise threshold (0.74% > 0.50%)
Pass: Cold: 1.181759ms GPU, 1.188029ms CPU, 14.59s total GPU, 12345x
Pass: Batch: 1.180483ms GPU, 14.57s total GPU, 12346x
Run:  copy_type_conversion_sweep [Device=0 In=I16 Out=I8]
Skip: Narrowing conversion: sizeof(InputType) > sizeof(OutputType).
Run:  copy_type_conversion_sweep [Device=0 In=I16 Out=I16]
Skip: Not a conversion: InputType == OutputType.
Run:  copy_type_conversion_sweep [Device=0 In=I16 Out=I32]
Warn: Current measurement timed out (15.00s) while over noise threshold (0.57% > 0.50%)
Pass: Cold: 0.447104ms GPU, 0.453377ms CPU, 13.72s total GPU, 30679x
Pass: Batch: 0.446394ms GPU, 13.70s total GPU, 30680x
Run:  copy_type_conversion_sweep [Device=0 In=I16 Out=F32]
Pass: Cold: 0.450117ms GPU, 0.456445ms CPU, 0.50s total GPU, 1111x
Pass: Batch: 0.447497ms GPU, 0.52s total GPU, 1162x
Run:  copy_type_conversion_sweep [Device=0 In=I16 Out=I64]
Warn: Current measurement timed out (15.00s) while over noise threshold (0.69% > 0.50%)
Pass: Cold: 0.657381ms GPU, 0.663639ms CPU, 14.19s total GPU, 21586x
Pass: Batch: 0.656117ms GPU, 14.16s total GPU, 21587x
Run:  copy_type_conversion_sweep [Device=0 In=I16 Out=F64]
Warn: Current measurement timed out (15.00s) while over noise threshold (0.70% > 0.50%)
Pass: Cold: 0.655457ms GPU, 0.661733ms CPU, 14.18s total GPU, 21638x
Pass: Batch: 0.653902ms GPU, 14.15s total GPU, 21639x
Run:  copy_type_conversion_sweep [Device=0 In=I32 Out=I8]
Skip: Narrowing conversion: sizeof(InputType) > sizeof(OutputType).
Run:  copy_type_conversion_sweep [Device=0 In=I32 Out=I16]
Skip: Narrowing conversion: sizeof(InputType) > sizeof(OutputType).
Run:  copy_type_conversion_sweep [Device=0 In=I32 Out=I32]
Skip: Not a conversion: InputType == OutputType.
Run:  copy_type_conversion_sweep [Device=0 In=I32 Out=F32]
Warn: Current measurement timed out (15.00s) while over noise threshold (1.26% > 0.50%)
Pass: Cold: 0.266290ms GPU, 0.272530ms CPU, 12.64s total GPU, 47462x
Pass: Batch: 0.264891ms GPU, 12.57s total GPU, 47463x
Run:  copy_type_conversion_sweep [Device=0 In=I32 Out=I64]
Warn: Current measurement timed out (15.00s) while over noise threshold (0.72% > 0.50%)
Pass: Cold: 0.378098ms GPU, 0.384345ms CPU, 13.43s total GPU, 35529x
Pass: Batch: 0.377321ms GPU, 13.41s total GPU, 35530x
Run:  copy_type_conversion_sweep [Device=0 In=I32 Out=F64]
Warn: Current measurement timed out (15.00s) while over noise threshold (0.81% > 0.50%)
Pass: Cold: 0.378571ms GPU, 0.384783ms CPU, 13.44s total GPU, 35498x
Pass: Batch: 0.377827ms GPU, 13.41s total GPU, 35499x
Run:  copy_type_conversion_sweep [Device=0 In=F32 Out=I8]
Skip: Narrowing conversion: sizeof(InputType) > sizeof(OutputType).
Run:  copy_type_conversion_sweep [Device=0 In=F32 Out=I16]
Skip: Narrowing conversion: sizeof(InputType) > sizeof(OutputType).
Run:  copy_type_conversion_sweep [Device=0 In=F32 Out=I32]
Warn: Current measurement timed out (15.00s) while over noise threshold (1.35% > 0.50%)
Pass: Cold: 0.265440ms GPU, 0.271689ms CPU, 12.64s total GPU, 47607x
Pass: Batch: 0.263872ms GPU, 12.56s total GPU, 47608x
Run:  copy_type_conversion_sweep [Device=0 In=F32 Out=F32]
Skip: Not a conversion: InputType == OutputType.
Run:  copy_type_conversion_sweep [Device=0 In=F32 Out=I64]
Warn: Current measurement timed out (15.00s) while over noise threshold (0.79% > 0.50%)
Pass: Cold: 0.378543ms GPU, 0.384786ms CPU, 13.44s total GPU, 35499x
Pass: Batch: 0.377921ms GPU, 13.42s total GPU, 35500x
Run:  copy_type_conversion_sweep [Device=0 In=F32 Out=F64]
Warn: Current measurement timed out (15.00s) while over noise threshold (0.84% > 0.50%)
Pass: Cold: 0.378288ms GPU, 0.384556ms CPU, 13.43s total GPU, 35509x
Pass: Batch: 0.377510ms GPU, 13.41s total GPU, 35510x
Run:  copy_type_conversion_sweep [Device=0 In=I64 Out=I8]
Skip: Narrowing conversion: sizeof(InputType) > sizeof(OutputType).
Run:  copy_type_conversion_sweep [Device=0 In=I64 Out=I16]
Skip: Narrowing conversion: sizeof(InputType) > sizeof(OutputType).
Run:  copy_type_conversion_sweep [Device=0 In=I64 Out=I32]
Skip: Narrowing conversion: sizeof(InputType) > sizeof(OutputType).
Run:  copy_type_conversion_sweep [Device=0 In=I64 Out=F32]
Skip: Narrowing conversion: sizeof(InputType) > sizeof(OutputType).
Run:  copy_type_conversion_sweep [Device=0 In=I64 Out=I64]
Skip: Not a conversion: InputType == OutputType.
Run:  copy_type_conversion_sweep [Device=0 In=I64 Out=F64]
Warn: Current measurement timed out (15.00s) while over noise threshold (0.88% > 0.50%)
Pass: Cold: 0.236185ms GPU, 0.242456ms CPU, 12.31s total GPU, 52100x
Pass: Batch: 0.235095ms GPU, 12.25s total GPU, 52101x
Run:  copy_type_conversion_sweep [Device=0 In=F64 Out=I8]
Skip: Narrowing conversion: sizeof(InputType) > sizeof(OutputType).
Run:  copy_type_conversion_sweep [Device=0 In=F64 Out=I16]
Skip: Narrowing conversion: sizeof(InputType) > sizeof(OutputType).
Run:  copy_type_conversion_sweep [Device=0 In=F64 Out=I32]
Skip: Narrowing conversion: sizeof(InputType) > sizeof(OutputType).
Run:  copy_type_conversion_sweep [Device=0 In=F64 Out=F32]
Skip: Narrowing conversion: sizeof(InputType) > sizeof(OutputType).
Run:  copy_type_conversion_sweep [Device=0 In=F64 Out=I64]
Warn: Current measurement timed out (15.00s) while over noise threshold (1.01% > 0.50%)
Pass: Cold: 0.238030ms GPU, 0.244316ms CPU, 12.33s total GPU, 51780x
Pass: Batch: 0.236852ms GPU, 12.26s total GPU, 51781x
Run:  copy_type_conversion_sweep [Device=0 In=F64 Out=F64]
Skip: Not a conversion: InputType == OutputType.
Run:  copy_type_conversion_sweep [Device=1 In=I8 Out=I8]
Skip: Not a conversion: InputType == OutputType.
Run:  copy_type_conversion_sweep [Device=1 In=I8 Out=I16]
Warn: Current measurement timed out (15.00s) while over noise threshold (0.80% > 0.50%)
Pass: Cold: 0.661753ms GPU, 0.666484ms CPU, 14.28s total GPU, 21576x
Pass: Batch: 0.660957ms GPU, 14.26s total GPU, 21577x
Run:  copy_type_conversion_sweep [Device=1 In=I8 Out=I32]
Warn: Current measurement timed out (15.00s) while over noise threshold (0.79% > 0.50%)
Pass: Cold: 0.858200ms GPU, 0.862918ms CPU, 14.47s total GPU, 16864x
Pass: Batch: 0.857568ms GPU, 14.46s total GPU, 16865x
Run:  copy_type_conversion_sweep [Device=1 In=I8 Out=F32]
Warn: Current measurement timed out (15.00s) while over noise threshold (0.80% > 0.50%)
Pass: Cold: 0.857883ms GPU, 0.862614ms CPU, 14.47s total GPU, 16866x
Pass: Batch: 0.857087ms GPU, 14.46s total GPU, 16867x
Run:  copy_type_conversion_sweep [Device=1 In=I8 Out=I64]
Warn: Current measurement timed out (15.00s) while over noise threshold (0.55% > 0.50%)
Pass: Cold: 1.455328ms GPU, 1.460033ms CPU, 14.72s total GPU, 10113x
Pass: Batch: 1.453767ms GPU, 14.70s total GPU, 10114x
Run:  copy_type_conversion_sweep [Device=1 In=I8 Out=F64]
Warn: Current measurement timed out (15.00s) while over noise threshold (0.55% > 0.50%)
Pass: Cold: 1.457198ms GPU, 1.461889ms CPU, 14.72s total GPU, 10100x
Pass: Batch: 1.455933ms GPU, 14.71s total GPU, 10101x
Run:  copy_type_conversion_sweep [Device=1 In=I16 Out=I8]
Skip: Narrowing conversion: sizeof(InputType) > sizeof(OutputType).
Run:  copy_type_conversion_sweep [Device=1 In=I16 Out=I16]
Skip: Not a conversion: InputType == OutputType.
Run:  copy_type_conversion_sweep [Device=1 In=I16 Out=I32]
Warn: Current measurement timed out (15.00s) while over noise threshold (0.79% > 0.50%)
Pass: Cold: 0.455669ms GPU, 0.460392ms CPU, 13.86s total GPU, 30414x
Pass: Batch: 0.455415ms GPU, 13.85s total GPU, 30415x
Run:  copy_type_conversion_sweep [Device=1 In=I16 Out=F32]
Warn: Current measurement timed out (15.00s) while over noise threshold (0.78% > 0.50%)
Pass: Cold: 0.454242ms GPU, 0.458932ms CPU, 13.86s total GPU, 30506x
Pass: Batch: 0.453806ms GPU, 13.84s total GPU, 30507x
Run:  copy_type_conversion_sweep [Device=1 In=I16 Out=I64]
Warn: Current measurement timed out (15.00s) while over noise threshold (0.56% > 0.50%)
Pass: Cold: 0.748931ms GPU, 0.753645ms CPU, 14.38s total GPU, 19198x
Pass: Batch: 0.748024ms GPU, 14.36s total GPU, 19199x
Run:  copy_type_conversion_sweep [Device=1 In=I16 Out=F64]
Warn: Current measurement timed out (15.00s) while over noise threshold (0.54% > 0.50%)
Pass: Cold: 0.747517ms GPU, 0.752207ms CPU, 14.38s total GPU, 19239x
Pass: Batch: 0.746459ms GPU, 14.36s total GPU, 19240x
Run:  copy_type_conversion_sweep [Device=1 In=I32 Out=I8]
Skip: Narrowing conversion: sizeof(InputType) > sizeof(OutputType).
Run:  copy_type_conversion_sweep [Device=1 In=I32 Out=I16]
Skip: Narrowing conversion: sizeof(InputType) > sizeof(OutputType).
Run:  copy_type_conversion_sweep [Device=1 In=I32 Out=I32]
Skip: Not a conversion: InputType == OutputType.
Run:  copy_type_conversion_sweep [Device=1 In=I32 Out=F32]
Warn: Current measurement timed out (15.00s) while over noise threshold (0.79% > 0.50%)
Pass: Cold: 0.273880ms GPU, 0.278585ms CPU, 12.87s total GPU, 47007x
Pass: Batch: 0.273539ms GPU, 12.86s total GPU, 47008x
Run:  copy_type_conversion_sweep [Device=1 In=I32 Out=I64]
Pass: Cold: 0.418105ms GPU, 0.422857ms CPU, 0.50s total GPU, 1196x
Pass: Batch: 0.416192ms GPU, 0.52s total GPU, 1254x
Run:  copy_type_conversion_sweep [Device=1 In=I32 Out=F64]
Pass: Cold: 0.418703ms GPU, 0.423383ms CPU, 0.50s total GPU, 1195x
Pass: Batch: 0.416603ms GPU, 0.52s total GPU, 1252x
Run:  copy_type_conversion_sweep [Device=1 In=F32 Out=I8]
Skip: Narrowing conversion: sizeof(InputType) > sizeof(OutputType).
Run:  copy_type_conversion_sweep [Device=1 In=F32 Out=I16]
Skip: Narrowing conversion: sizeof(InputType) > sizeof(OutputType).
Run:  copy_type_conversion_sweep [Device=1 In=F32 Out=I32]
Warn: Current measurement timed out (15.00s) while over noise threshold (1.25% > 0.50%)
Pass: Cold: 0.276847ms GPU, 0.281581ms CPU, 12.89s total GPU, 46545x
Pass: Batch: 0.276479ms GPU, 12.87s total GPU, 46546x
Run:  copy_type_conversion_sweep [Device=1 In=F32 Out=F32]
Skip: Not a conversion: InputType == OutputType.
Run:  copy_type_conversion_sweep [Device=1 In=F32 Out=I64]
Pass: Cold: 0.418391ms GPU, 0.423070ms CPU, 0.50s total GPU, 1196x
Pass: Batch: 0.416373ms GPU, 0.52s total GPU, 1257x
Run:  copy_type_conversion_sweep [Device=1 In=F32 Out=F64]
Pass: Cold: 0.418690ms GPU, 0.423378ms CPU, 0.50s total GPU, 1195x
Pass: Batch: 0.416660ms GPU, 0.53s total GPU, 1265x
Run:  copy_type_conversion_sweep [Device=1 In=I64 Out=I8]
Skip: Narrowing conversion: sizeof(InputType) > sizeof(OutputType).
Run:  copy_type_conversion_sweep [Device=1 In=I64 Out=I16]
Skip: Narrowing conversion: sizeof(InputType) > sizeof(OutputType).
Run:  copy_type_conversion_sweep [Device=1 In=I64 Out=I32]
Skip: Narrowing conversion: sizeof(InputType) > sizeof(OutputType).
Run:  copy_type_conversion_sweep [Device=1 In=I64 Out=F32]
Skip: Narrowing conversion: sizeof(InputType) > sizeof(OutputType).
Run:  copy_type_conversion_sweep [Device=1 In=I64 Out=I64]
Skip: Not a conversion: InputType == OutputType.
Run:  copy_type_conversion_sweep [Device=1 In=I64 Out=F64]
Pass: Cold: 0.261885ms GPU, 0.266569ms CPU, 0.50s total GPU, 1910x
Pass: Batch: 0.260037ms GPU, 0.52s total GPU, 2011x
Run:  copy_type_conversion_sweep [Device=1 In=F64 Out=I8]
Skip: Narrowing conversion: sizeof(InputType) > sizeof(OutputType).
Run:  copy_type_conversion_sweep [Device=1 In=F64 Out=I16]
Skip: Narrowing conversion: sizeof(InputType) > sizeof(OutputType).
Run:  copy_type_conversion_sweep [Device=1 In=F64 Out=I32]
Skip: Narrowing conversion: sizeof(InputType) > sizeof(OutputType).
Run:  copy_type_conversion_sweep [Device=1 In=F64 Out=F32]
Skip: Narrowing conversion: sizeof(InputType) > sizeof(OutputType).
Run:  copy_type_conversion_sweep [Device=1 In=F64 Out=I64]
Pass: Cold: 0.261586ms GPU, 0.266286ms CPU, 0.50s total GPU, 1912x
Pass: Batch: 0.259790ms GPU, 0.52s total GPU, 2016x
Run:  copy_type_conversion_sweep [Device=1 In=F64 Out=F64]
Skip: Not a conversion: InputType == OutputType.
```

# Benchmark Results

## simple

### [0] Quadro GV100

| Samples | CPU Time | Noise | GPU Time | Noise | Batch GPU | Batch |
|---------|----------|-------|----------|-------|-----------|-------|
|    499x | 1.010 ms | 0.05% | 1.004 ms | 0.06% |  1.001 ms |  524x |

### [1] Quadro GP100

| Samples | CPU Time | Noise | GPU Time | Noise | Batch GPU | Batch |
|---------|----------|-------|----------|-------|-----------|-------|
|    499x | 1.007 ms | 0.04% | 1.003 ms | 0.03% |  1.001 ms |  524x |

## single_float64_axis

### [0] Quadro GV100

| Duration | Samples |  CPU Time  | Noise |  GPU Time  | Noise  | Batch GPU  |  Batch  |
|----------|---------|------------|-------|------------|--------|------------|---------|
|        0 | 147957x |  10.618 us | 3.25% |   4.424 us | 10.78% |   2.043 us | 244766x |
|   0.0001 |   4831x | 110.048 us | 0.42% | 103.515 us |  0.48% | 101.376 us |   5088x |
|   0.0002 |   2453x | 210.369 us | 0.22% | 203.903 us |  0.25% | 201.729 us |   2582x |
|   0.0003 |   1648x | 309.866 us | 0.15% | 303.412 us |  0.17% | 301.164 us |   1736x |
|   0.0004 |   1239x | 410.148 us | 0.12% | 403.673 us |  0.14% | 401.410 us |   1304x |
|   0.0005 |    992x | 510.529 us | 0.09% | 504.089 us |  0.11% | 501.762 us |   1042x |
|   0.0006 |    829x | 609.862 us | 0.08% | 603.471 us |  0.10% | 601.104 us |    872x |
|   0.0007 |    711x | 710.294 us | 0.07% | 703.744 us |  0.08% | 701.443 us |    748x |
|   0.0008 |    622x | 810.565 us | 0.06% | 804.187 us |  0.07% | 801.795 us |    653x |
|   0.0009 |    554x | 909.873 us | 0.05% | 903.433 us |  0.06% | 901.125 us |    582x |
|    0.001 |    499x |   1.010 ms | 0.04% |   1.004 ms |  0.05% |   1.001 ms |    523x |

### [1] Quadro GP100

| Duration | Samples |  CPU Time  | Noise |  GPU Time  | Noise | Batch GPU  |  Batch  |
|----------|---------|------------|-------|------------|-------|------------|---------|
|        0 | 152839x |   7.705 us | 5.42% |   3.016 us | 4.13% |   1.343 us | 372166x |
|   0.0001 |   4879x | 107.156 us | 0.41% | 102.481 us | 0.31% | 101.376 us |   5107x |
|   0.0002 |   2466x | 207.544 us | 0.19% | 202.833 us | 0.15% | 201.728 us |   2586x |
|   0.0003 |   1655x | 306.880 us | 0.13% | 302.191 us | 0.11% | 301.057 us |   1736x |
|   0.0004 |   1243x | 407.214 us | 0.11% | 402.508 us | 0.08% | 401.409 us |   1305x |
|   0.0005 |    995x | 507.562 us | 0.08% | 502.864 us | 0.06% | 501.761 us |   1045x |
|   0.0006 |    831x | 606.954 us | 0.07% | 602.223 us | 0.05% | 601.089 us |    873x |
|   0.0007 |    712x | 707.255 us | 0.06% | 702.559 us | 0.04% | 701.442 us |    748x |
|   0.0008 |    623x | 807.636 us | 0.05% | 802.910 us | 0.04% | 801.794 us |    655x |
|   0.0009 |    555x | 906.935 us | 0.05% | 902.248 us | 0.03% | 901.123 us |    582x |
|    0.001 |    499x |   1.007 ms | 0.04% |   1.003 ms | 0.03% |   1.001 ms |    524x |

## copy_sweep_grid_shape

### [0] Quadro GV100

| BlockSize | (BlockSize) | NumBlocks | (NumBlocks) | Samples |  CPU Time  | Noise |  GPU Time  | Noise | Elem/s  | GlobalMem BW | BWPeak | Batch GPU  | Batch  |
|-----------|-------------|-----------|-------------|---------|------------|-------|------------|-------|---------|--------------|--------|------------|--------|
|       2^6 |          64 |       2^6 |          64 |     66x |   7.622 ms | 0.14% |   7.616 ms | 0.14% |  8.812G |  70.495 GB/s |  8.10% |   7.615 ms |    69x |
|       2^8 |         256 |       2^6 |          64 |    206x |   2.442 ms | 0.46% |   2.436 ms | 0.46% | 27.547G | 220.379 GB/s | 25.32% |   2.434 ms |   215x |
|      2^10 |        1024 |       2^6 |          64 |  13161x |   1.112 ms | 1.29% |   1.106 ms | 1.30% | 60.681G | 485.445 GB/s | 55.77% |   1.103 ms | 13162x |
|       2^6 |          64 |       2^8 |         256 |    375x |   2.450 ms | 0.50% |   2.444 ms | 0.50% | 27.457G | 219.652 GB/s | 25.24% |   2.444 ms |   376x |
|       2^8 |         256 |       2^8 |         256 |  13509x |   1.083 ms | 0.96% |   1.077 ms | 0.97% | 62.332G | 498.653 GB/s | 57.29% |   1.076 ms | 13510x |
|      2^10 |        1024 |       2^8 |         256 |  15105x | 964.751 us | 0.51% | 958.478 us | 0.51% | 70.016G | 560.129 GB/s | 64.35% | 957.249 us | 15106x |
|       2^6 |          64 |      2^10 |        1024 |  13582x |   1.077 ms | 0.74% |   1.071 ms | 0.74% | 62.682G | 501.460 GB/s | 57.61% |   1.071 ms | 13583x |
|       2^8 |         256 |      2^10 |        1024 |   1782x | 962.899 us | 0.50% | 956.568 us | 0.50% | 70.156G | 561.247 GB/s | 64.48% | 954.599 us |  1783x |
|      2^10 |        1024 |      2^10 |        1024 |  14579x |   1.000 ms | 1.93% | 994.218 us | 1.94% | 67.499G | 539.993 GB/s | 62.04% | 992.819 us | 14580x |

### [1] Quadro GP100

| BlockSize | (BlockSize) | NumBlocks | (NumBlocks) | Samples | CPU Time | Noise | GPU Time | Noise | Elem/s  | GlobalMem BW | BWPeak | Batch GPU | Batch  |
|-----------|-------------|-----------|-------------|---------|----------|-------|----------|-------|---------|--------------|--------|-----------|--------|
|       2^6 |          64 |       2^6 |          64 |   2236x | 6.689 ms | 1.09% | 6.684 ms | 1.10% | 10.040G |  80.319 GB/s | 10.97% |  6.675 ms |  2237x |
|       2^8 |         256 |       2^6 |          64 |    218x | 2.301 ms | 0.29% | 2.296 ms | 0.29% | 29.224G | 233.794 GB/s | 31.93% |  2.298 ms |   228x |
|      2^10 |        1024 |       2^6 |          64 |    426x | 1.179 ms | 0.39% | 1.174 ms | 0.39% | 57.144G | 457.155 GB/s | 62.44% |  1.172 ms |   449x |
|       2^6 |          64 |       2^8 |         256 |    226x | 2.218 ms | 0.16% | 2.214 ms | 0.16% | 30.316G | 242.531 GB/s | 33.13% |  2.213 ms |   237x |
|       2^8 |         256 |       2^8 |         256 |  12933x | 1.135 ms | 0.67% | 1.131 ms | 0.67% | 59.361G | 474.891 GB/s | 64.86% |  1.130 ms | 12934x |
|      2^10 |        1024 |       2^8 |         256 |    447x | 1.124 ms | 0.22% | 1.119 ms | 0.22% | 59.975G | 479.797 GB/s | 65.53% |  1.117 ms |   468x |
|       2^6 |          64 |      2^10 |        1024 |    448x | 1.122 ms | 0.30% | 1.117 ms | 0.30% | 60.084G | 480.669 GB/s | 65.65% |  1.115 ms |   470x |
|       2^8 |         256 |      2^10 |        1024 |    448x | 1.122 ms | 0.28% | 1.118 ms | 0.28% | 60.042G | 480.335 GB/s | 65.61% |  1.116 ms |   471x |
|      2^10 |        1024 |      2^10 |        1024 |    474x | 1.060 ms | 0.15% | 1.056 ms | 0.15% | 63.568G | 508.542 GB/s | 69.46% |  1.054 ms |   498x |

## copy_type_sweep

### [0] Quadro GV100

|  T  | Samples |  CPU Time  | Noise |  GPU Time  | Noise |  Elem/s  | GlobalMem BW | BWPeak | Batch GPU  | Batch  |
|-----|---------|------------|-------|------------|-------|----------|--------------|--------|------------|--------|
|  U8 |    197x |   2.550 ms | 0.30% |   2.544 ms | 0.30% | 105.536G | 211.072 GB/s | 24.25% |   2.539 ms |   206x |
| U16 |    314x |   1.602 ms | 0.41% |   1.596 ms | 0.41% |  84.116G | 336.465 GB/s | 38.66% |   1.592 ms |   331x |
| U32 |  13509x |   1.083 ms | 0.96% |   1.077 ms | 0.97% |  62.333G | 498.668 GB/s | 57.29% |   1.075 ms | 13510x |
| U64 |  15542x | 936.680 us | 0.60% | 930.416 us | 0.61% |  36.064G | 577.023 GB/s | 66.29% | 929.189 us | 15543x |
| F32 |  13508x |   1.083 ms | 0.97% |   1.077 ms | 0.97% |  62.323G | 498.586 GB/s | 57.28% |   1.075 ms | 13509x |
| F64 |  15546x | 936.683 us | 0.59% | 930.412 us | 0.60% |  36.064G | 577.025 GB/s | 66.29% | 929.182 us | 15547x |

### [1] Quadro GP100

|  T  | Samples | CPU Time | Noise | GPU Time | Noise | Elem/s  | GlobalMem BW | BWPeak | Batch GPU | Batch  |
|-----|---------|----------|-------|----------|-------|---------|--------------|--------|-----------|--------|
|  U8 |   5497x | 2.708 ms | 0.64% | 2.703 ms | 0.64% | 99.307G | 198.614 GB/s | 27.13% |  2.700 ms |  5498x |
| U16 |    330x | 1.520 ms | 0.44% | 1.515 ms | 0.44% | 88.573G | 354.292 GB/s | 48.39% |  1.514 ms |   348x |
| U32 |  12935x | 1.135 ms | 0.66% | 1.131 ms | 0.66% | 59.356G | 474.846 GB/s | 64.86% |  1.130 ms | 12936x |
| U64 |    478x | 1.052 ms | 0.27% | 1.048 ms | 0.27% | 32.032G | 512.520 GB/s | 70.00% |  1.045 ms |   500x |
| F32 |  12933x | 1.135 ms | 0.67% | 1.131 ms | 0.67% | 59.355G | 474.842 GB/s | 64.85% |  1.130 ms | 12934x |
| F64 |    477x | 1.053 ms | 0.28% | 1.048 ms | 0.28% | 32.005G | 512.078 GB/s | 69.94% |  1.046 ms |   497x |

## copy_type_conversion_sweep

### [0] Quadro GV100

| In  | Out |  Items   |   InSize   |   OutSize   | Samples |  CPU Time  | Noise |  GPU Time  | Noise | Elem/s  | GlobalMem BW | BWPeak | Batch GPU  | Batch  |
|-----|-----|----------|------------|-------------|---------|------------|-------|------------|-------|---------|--------------|--------|------------|--------|
|  I8 | I16 | 67108864 | 64.000 MiB | 128.000 MiB |    712x | 709.177 us | 0.29% | 702.933 us | 0.30% | 95.470G | 286.410 GB/s | 32.91% | 699.714 us |   748x |
|  I8 | I32 | 67108864 | 64.000 MiB | 256.000 MiB |    622x | 810.934 us | 0.31% | 804.698 us | 0.31% | 83.396G | 416.982 GB/s | 47.91% | 802.089 us |   654x |
|  I8 | F32 | 67108864 | 64.000 MiB | 256.000 MiB |    614x | 821.028 us | 0.32% | 814.768 us | 0.33% | 82.366G | 411.828 GB/s | 47.31% | 812.088 us |   645x |
|  I8 | I64 | 67108864 | 64.000 MiB | 512.000 MiB |  12047x |   1.218 ms | 1.04% |   1.212 ms | 1.04% | 55.369G | 498.319 GB/s | 57.25% |   1.211 ms | 12048x |
|  I8 | F64 | 67108864 | 64.000 MiB | 512.000 MiB |  12345x |   1.188 ms | 0.74% |   1.182 ms | 0.74% | 56.787G | 511.086 GB/s | 58.72% |   1.180 ms | 12346x |
| I16 | I32 | 33554432 | 64.000 MiB | 128.000 MiB |  30679x | 453.377 us | 0.56% | 447.104 us | 0.57% | 75.048G | 450.290 GB/s | 51.73% | 446.394 us | 30680x |
| I16 | F32 | 33554432 | 64.000 MiB | 128.000 MiB |   1111x | 456.445 us | 0.46% | 450.117 us | 0.47% | 74.546G | 447.276 GB/s | 51.39% | 447.497 us |  1162x |
| I16 | I64 | 33554432 | 64.000 MiB | 256.000 MiB |  21586x | 663.639 us | 0.68% | 657.381 us | 0.69% | 51.043G | 510.426 GB/s | 58.64% | 656.117 us | 21587x |
| I16 | F64 | 33554432 | 64.000 MiB | 256.000 MiB |  21638x | 661.733 us | 0.69% | 655.457 us | 0.70% | 51.192G | 511.925 GB/s | 58.81% | 653.902 us | 21639x |
| I32 | F32 | 16777216 | 64.000 MiB |  64.000 MiB |  47462x | 272.530 us | 1.23% | 266.290 us | 1.26% | 63.003G | 504.028 GB/s | 57.91% | 264.891 us | 47463x |
| I32 | I64 | 16777216 | 64.000 MiB | 128.000 MiB |  35529x | 384.345 us | 0.71% | 378.098 us | 0.72% | 44.373G | 532.471 GB/s | 61.18% | 377.321 us | 35530x |
| I32 | F64 | 16777216 | 64.000 MiB | 128.000 MiB |  35498x | 384.783 us | 0.79% | 378.571 us | 0.81% | 44.317G | 531.807 GB/s | 61.10% | 377.827 us | 35499x |
| F32 | I32 | 16777216 | 64.000 MiB |  64.000 MiB |  47607x | 271.689 us | 1.32% | 265.440 us | 1.35% | 63.205G | 505.642 GB/s | 58.09% | 263.872 us | 47608x |
| F32 | I64 | 16777216 | 64.000 MiB | 128.000 MiB |  35499x | 384.786 us | 0.78% | 378.543 us | 0.79% | 44.320G | 531.846 GB/s | 61.10% | 377.921 us | 35500x |
| F32 | F64 | 16777216 | 64.000 MiB | 128.000 MiB |  35509x | 384.556 us | 0.82% | 378.288 us | 0.84% | 44.350G | 532.204 GB/s | 61.14% | 377.510 us | 35510x |
| I64 | F64 |  8388608 | 64.000 MiB |  64.000 MiB |  52100x | 242.456 us | 0.85% | 236.185 us | 0.88% | 35.517G | 568.273 GB/s | 65.29% | 235.095 us | 52101x |
| F64 | I64 |  8388608 | 64.000 MiB |  64.000 MiB |  51780x | 244.316 us | 0.98% | 238.030 us | 1.01% | 35.242G | 563.869 GB/s | 64.78% | 236.852 us | 51781x |

### [1] Quadro GP100

| In  | Out |  Items   |   InSize   |   OutSize   | Samples |  CPU Time  | Noise |  GPU Time  | Noise |  Elem/s  | GlobalMem BW | BWPeak | Batch GPU  | Batch  |
|-----|-----|----------|------------|-------------|---------|------------|-------|------------|-------|----------|--------------|--------|------------|--------|
|  I8 | I16 | 67108864 | 64.000 MiB | 128.000 MiB |  21576x | 666.484 us | 0.80% | 661.753 us | 0.80% | 101.411G | 304.232 GB/s | 41.55% | 660.957 us | 21577x |
|  I8 | I32 | 67108864 | 64.000 MiB | 256.000 MiB |  16864x | 862.918 us | 0.79% | 858.200 us | 0.79% |  78.197G | 390.986 GB/s | 53.40% | 857.568 us | 16865x |
|  I8 | F32 | 67108864 | 64.000 MiB | 256.000 MiB |  16866x | 862.614 us | 0.80% | 857.883 us | 0.80% |  78.226G | 391.131 GB/s | 53.42% | 857.087 us | 16867x |
|  I8 | I64 | 67108864 | 64.000 MiB | 512.000 MiB |  10113x |   1.460 ms | 0.55% |   1.455 ms | 0.55% |  46.113G | 415.013 GB/s | 56.68% |   1.454 ms | 10114x |
|  I8 | F64 | 67108864 | 64.000 MiB | 512.000 MiB |  10100x |   1.462 ms | 0.54% |   1.457 ms | 0.55% |  46.053G | 414.480 GB/s | 56.61% |   1.456 ms | 10101x |
| I16 | I32 | 33554432 | 64.000 MiB | 128.000 MiB |  30414x | 460.392 us | 0.78% | 455.669 us | 0.79% |  73.638G | 441.826 GB/s | 60.35% | 455.415 us | 30415x |
| I16 | F32 | 33554432 | 64.000 MiB | 128.000 MiB |  30506x | 458.932 us | 0.78% | 454.242 us | 0.78% |  73.869G | 443.215 GB/s | 60.54% | 453.806 us | 30507x |
| I16 | I64 | 33554432 | 64.000 MiB | 256.000 MiB |  19198x | 753.645 us | 0.56% | 748.931 us | 0.56% |  44.803G | 448.031 GB/s | 61.19% | 748.024 us | 19199x |
| I16 | F64 | 33554432 | 64.000 MiB | 256.000 MiB |  19239x | 752.207 us | 0.54% | 747.517 us | 0.54% |  44.888G | 448.878 GB/s | 61.31% | 746.459 us | 19240x |
| I32 | F32 | 16777216 | 64.000 MiB |  64.000 MiB |  47007x | 278.585 us | 0.78% | 273.880 us | 0.79% |  61.258G | 490.060 GB/s | 66.93% | 273.539 us | 47008x |
| I32 | I64 | 16777216 | 64.000 MiB | 128.000 MiB |   1196x | 422.857 us | 0.44% | 418.105 us | 0.44% |  40.127G | 481.521 GB/s | 65.77% | 416.192 us |  1254x |
| I32 | F64 | 16777216 | 64.000 MiB | 128.000 MiB |   1195x | 423.383 us | 0.47% | 418.703 us | 0.47% |  40.070G | 480.834 GB/s | 65.67% | 416.603 us |  1252x |
| F32 | I32 | 16777216 | 64.000 MiB |  64.000 MiB |  46545x | 281.581 us | 1.23% | 276.847 us | 1.25% |  60.601G | 484.808 GB/s | 66.22% | 276.479 us | 46546x |
| F32 | I64 | 16777216 | 64.000 MiB | 128.000 MiB |   1196x | 423.070 us | 0.46% | 418.391 us | 0.46% |  40.099G | 481.193 GB/s | 65.72% | 416.373 us |  1257x |
| F32 | F64 | 16777216 | 64.000 MiB | 128.000 MiB |   1195x | 423.378 us | 0.47% | 418.690 us | 0.47% |  40.071G | 480.849 GB/s | 65.68% | 416.660 us |  1265x |
| I64 | F64 |  8388608 | 64.000 MiB |  64.000 MiB |   1910x | 266.569 us | 0.42% | 261.885 us | 0.42% |  32.032G | 512.506 GB/s | 70.00% | 260.037 us |  2011x |
| F64 | I64 |  8388608 | 64.000 MiB |  64.000 MiB |   1912x | 266.286 us | 0.42% | 261.586 us | 0.41% |  32.068G | 513.092 GB/s | 70.08% | 259.790 us |  2016x |
