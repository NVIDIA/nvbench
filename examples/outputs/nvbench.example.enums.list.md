# Devices

## [0] `Quadro GV100`
* SM Version: 700 (PTX Version: 700)
* Number of SMs: 80
* SM Default Clock Rate: 1627 MHz
* Global Memory: 30117 MiB Free / 32507 MiB Total
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
* Global Memory: 14939 MiB Free / 16278 MiB Total
* Global Memory Bus Peak: 732 GB/sec (4096-bit DDR @715MHz)
* Max Shared Memory: 64 KiB/SM, 48 KiB/Block
* L2 Cache Size: 4096 KiB
* Maximum Active Blocks: 32/SM
* Maximum Active Threads: 2048/SM, 1024/Block
* Available Registers: 65536/SM, 65536/Block
* ECC Enabled: No

# Benchmarks

## [0] `runtime_enum_sweep_string` (3 configurations)

### Axes

* `MyEnum` : string
  * `A`
  * `B`
  * `C`

## [1] `runtime_enum_sweep_int64` (3 configurations)

### Axes

* `MyEnum` : int64
  * `0`
  * `1`
  * `2`

## [2] `compile_time_enum_sweep` (3 configurations)

### Axes

* `MyEnum` : type
  * `A` (MyEnum::ValueA)
  * `B` (MyEnum::ValueB)
  * `C` (MyEnum::ValueC)

## [3] `compile_time_int_sweep` (4 configurations)

### Axes

* `SomeInts` : type
  * `0` (nvbench::enum_type<0, int>)
  * `16` (nvbench::enum_type<16, int>)
  * `4096` (nvbench::enum_type<4096, int>)
  * `-12` (nvbench::enum_type<-12, int>)

