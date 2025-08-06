# CUDA Kernel Benchmarking Package

This package provides Python API to CUDA Kernel Benchmarking Library `NVBench`.

## Building

### Ensure recent version of CMake

Since `nvbench` requires a rather new version of CMake (>=3.30.4), either build CMake from sources, or create a conda environment with a recent version of CMake, using

```
conda create -n build_env --yes  cmake ninja
conda activate build_env
```

### Ensure CUDA compiler

Since building `NVBench` library requires CUDA compiler, ensure that appropriate environment variables
are set. For example, assuming CUDA toolkit is installed system-wide, and assuming Ampere GPU architecture:

```bash
export CUDACXX=/usr/local/cuda/bin/nvcc
export CUDAARCHS=86
``

### Build Python project

Now switch to python folder, configure and install NVBench library, and install the package in editable mode:

```bash
cd nvbench/python
pip install -e .
```

### Verify that package works

```bash
python test/run_1.py
```

### Run examples

```bash
# Example benchmarking numba.cuda kernel
python examples/throughput.py
```

```bash
# Example benchmarking kernels authored using cuda.core
python examples/axes.py
```

```bash
# Example benchmarking algorithms from cuda.cccl.parallel
python examples/cccl_parallel_segmented_reduce.py
```

```bash
# Example benchmarking CuPy function
python examples/cupy_extract.py
```
