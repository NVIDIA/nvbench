# CUDA Kernel Benchmarking Package

This package provides Python API to CUDA Kernel Benchmarking Library `NVBench`.

## Building

### Build `NVBench` project

Since `nvbench` requires a rather new version of CMake (>=3.30.4), either build CMake from sources, or create a conda environment with a recent version of CMake, using

```
conda create -n build_env --yes  cmake ninja
conda activate build_env
```

Now switch to python folder, configure and install NVBench library, and install the package in editable mode:

```
cd nvbench/python
cmake -B nvbench_build --preset nvbench-ci -S $(pwd)/.. -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DNVBench_ENABLE_EXAMPLES=OFF -DCMAKE_INSTALL_PREFIX=$(pwd)/nvbench_install
cmake --build nvbench_build/ --config Release --target install

nvbench_DIR=$(pwd)/nvbench_install/lib/cmake CUDACXX=/usr/local/cuda/bin/nvcc pip install -e .
```

### Verify that package works

```
export PYTHONPATH=$(pwd):${PYTHONPATH}
python test/run_1.py
```

### Run examples

```
# Example benchmarking numba.cuda kernel
python examples/throughput.py
```

```
# Example benchmarking kernels authored using cuda.core
python examples/axes.py
```

```
# Example benchmarking algorithms from cuda.cccl.parallel
python examples/cccl_parallel_segmented_reduce.py
```

```
# Example benchmarking CuPy function
python examples/cupy_extract.py
```
