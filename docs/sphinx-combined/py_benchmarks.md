# NVBench: benchmarking in Python

The `cuda.bench` Python module provides Python API powered by C++ NVBench
library to benchmark GPU-aware Python code.

## Minimal benchmark

```python
from cuda.bench import State, Launch
from cuda.bench import register, axis, options, run_all_registered
from typing import Callable

from my_package import impl, generate

@register()
@options.set_name("my_package_kernel")
@axis.add_int64_axis("Elements", [1000, 10000, 100000])
def benchmark_impl(state: State) -> None:

    # get state parameters
    n = state.get_int64("Elements")

    # prepare inputs
    data = generate(n, state.get_stream())

    # body that is being timed. Must execute
    # on the stream handed over by NVBench.
    # Typically launches a kernel of interest
    launch_fn : Callable[[Launch], None] =
       lambda launch: impl(data, launch.get_stream())

    state.exec(launch_fn)


if __name__ == "__main__":
   import sys
   run_all_registered(sys.argv)
```
