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
* Global Memory: 14891 MiB Free / 16278 MiB Total
* Global Memory Bus Peak: 732 GB/sec (4096-bit DDR @715MHz)
* Max Shared Memory: 64 KiB/SM, 48 KiB/Block
* L2 Cache Size: 4096 KiB
* Maximum Active Blocks: 32/SM
* Maximum Active Threads: 2048/SM, 1024/Block
* Available Registers: 65536/SM, 65536/Block
* ECC Enabled: No

# Benchmarks

## [0] `simple` (1 configurations)

## [1] `single_float64_axis` (11 configurations)

### Axes

* `Duration` : float64
  * `0`
  * `0.0001`
  * `0.0002`
  * `0.0003`
  * `0.0004`
  * `0.0005`
  * `0.0006`
  * `0.0007`
  * `0.0008`
  * `0.0009`
  * `0.001`

## [2] `copy_sweep_grid_shape` (9 configurations)

### Axes

* `BlockSize` : int64 [pow2]
  * `6` (2^6 = 64)
  * `8` (2^8 = 256)
  * `10` (2^10 = 1024)
* `NumBlocks` : int64 [pow2]
  * `6` (2^6 = 64)
  * `8` (2^8 = 256)
  * `10` (2^10 = 1024)

## [3] `copy_type_sweep` (6 configurations)

### Axes

* `T` : type
  * `U8` (uint8_t)
  * `U16` (uint16_t)
  * `U32` (uint32_t)
  * `U64` (uint64_t)
  * `F32` (float)
  * `F64` (double)

## [4] `copy_type_conversion_sweep` (36 configurations)

### Axes

* `In` : type
  * `I8` (int8_t)
  * `I16` (int16_t)
  * `I32` (int32_t)
  * `F32` (float)
  * `I64` (int64_t)
  * `F64` (double)
* `Out` : type
  * `I8` (int8_t)
  * `I16` (int16_t)
  * `I32` (int32_t)
  * `F32` (float)
  * `I64` (int64_t)
  * `F64` (double)

