# CUDA Kernel Benchmarking Package

This package provides a Python API to the CUDA Kernel Benchmarking
Library `NVBench`.

## Installation

Install from PyPI:

```bash
python -m pip install cuda-bench
```

Use an optional dependency if you want `pip` to install a compatible
`cuda-bindings` package as well:

```bash
python -m pip install "cuda-bench[cu12]"  # Install cuda-bindings 12.x
python -m pip install "cuda-bench[cu13]"  # Install cuda-bindings 13.x
```

The published Linux wheel is compatible with both CUDA 12.x and CUDA 13.x
Python environments. It contains two native extensions: one built with a CUDA
12.x Toolkit and installed under `cuda.bench.cu12`, and one built with a CUDA
13.x Toolkit and installed under `cuda.bench.cu13`. At runtime, `cuda-bench`
queries the installed `cuda.bindings` package to determine the CUDA major
version and loads the matching native extension.

The `cu12` and `cu13` extras do not select different `cuda-bench` wheels. They
only select the compatible `cuda-bindings` dependency family. If your
environment already provides an appropriate `cuda-bindings` 12.x or 13.x
package, installing plain `cuda-bench` is sufficient.

A local CUDA Toolkit is not required when installing a published wheel, but the
NVIDIA driver must support the CUDA runtime used by the installed
`cuda.bindings` package. Use the same CUDA major version for other CUDA Python
binary packages in the environment, for example `cupy-cuda12x` with
`cuda-bench[cu12]` or `cupy-cuda13x` with `cuda-bench[cu13]`.

## Building from source

### Ensure recent version of CMake

Since `nvbench` requires CMake >=3.30.4, either install a recent CMake or
create a conda environment with CMake and Ninja:

```bash
conda create -n build_env --yes cmake ninja
conda activate build_env
```

### Ensure CUDA compiler

Building `cuda-bench` from source requires a CUDA Toolkit with `nvcc`. Ensure
that the appropriate environment variables are set. For example, on Linux,
assuming the CUDA Toolkit is installed system-wide:

```bash
export CUDACXX=/usr/local/cuda/bin/nvcc
export CUDAARCHS=all-major
```

Unlike the published wheel, a local source build only builds the native
extension for the CUDA Toolkit found by CMake. The CUDA major version selected
in the install command below must match that Toolkit.

### Build Python project

Now switch to the Python package directory and install `cuda-bench` from source:

```bash
cd nvbench/python
python -m pip install ".[cu12]"  # If CUDACXX points to a CUDA 12.x toolkit
python -m pip install ".[cu13]"  # If CUDACXX points to a CUDA 13.x toolkit
```

Editable installs (`python -m pip install -e .`) are currently not supported.
They do not install the versioned CUDA extension layout used by `cuda-bench`.
Re-run the non-editable install command after making source changes.

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
