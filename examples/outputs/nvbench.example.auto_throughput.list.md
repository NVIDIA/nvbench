# Devices

## [0] `Quadro GV100`
* SM Version: 700 (PTX Version: 700)
* Number of SMs: 80
* SM Default Clock Rate: 1627 MHz
* Global Memory: 31601 MiB Free / 32507 MiB Total
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
* Global Memory: 15563 MiB Free / 16278 MiB Total
* Global Memory Bus Peak: 732 GB/sec (4096-bit DDR @715MHz)
* Max Shared Memory: 64 KiB/SM, 48 KiB/Block
* L2 Cache Size: 4096 KiB
* Maximum Active Blocks: 32/SM
* Maximum Active Threads: 2048/SM, 1024/Block
* Available Registers: 65536/SM, 65536/Block
* ECC Enabled: No

# Benchmarks

## [0] `throughput_bench` (4 configurations)

### Axes

* `T` : type
  * `1` (nvbench::enum_type<1, int>)
  * `2` (nvbench::enum_type<2, int>)
* `Stride` : int64
  * `1`
  * `4`

