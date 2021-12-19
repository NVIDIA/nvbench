# Devices

## [0] `Quadro GV100`
* SM Version: 700 (PTX Version: 700)
* Number of SMs: 80
* SM Default Clock Rate: 1627 MHz
* Global Memory: 31309 MiB Free / 32507 MiB Total
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
* Global Memory: 15467 MiB Free / 16278 MiB Total
* Global Memory Bus Peak: 732 GB/sec (4096-bit DDR @715MHz)
* Max Shared Memory: 64 KiB/SM, 48 KiB/Block
* L2 Cache Size: 4096 KiB
* Maximum Active Blocks: 32/SM
* Maximum Active Threads: 2048/SM, 1024/Block
* Available Registers: 65536/SM, 65536/Block
* ECC Enabled: No

# Benchmarks

## [0] `runtime_skip` (15 configurations)

### Axes

* `Duration` : float64
  * `0`
  * `0.00025`
  * `0.0005`
  * `0.00075`
  * `0.001`
* `Kramble` : string
  * `Foo`
  * `Bar`
  * `Baz`

## [1] `skip_overload` (4 configurations)

### Axes

* `In` : type
  * `I32` (int32_t)
  * `I64` (int64_t)
* `Out` : type
  * `I32` (int32_t)
  * `I64` (int64_t)

## [2] `skip_sfinae` (16 configurations)

### Axes

* `In` : type
  * `I8` (int8_t)
  * `I16` (int16_t)
  * `I32` (int32_t)
  * `I64` (int64_t)
* `Out` : type
  * `I8` (int8_t)
  * `I16` (int16_t)
  * `I32` (int32_t)
  * `I64` (int64_t)

