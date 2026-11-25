CUDA Kernel Benchmarking Library
================================

The library, NVBench, presently supports writing benchmarks in C++ and in Python.
It is designed to simplify CUDA kernel benchmarking. It features:

* :ref:`Parameter sweeps <parameter-axes>`: a powerful and
  flexible "axis" system explores a kernel's configuration space. Parameters may
  be dynamic numbers/strings or :ref:`static types <type-axes>`.
* :ref:`Runtime customization <cli-overview>`: A rich command-line interface
  allows :ref:`redefinition of parameter axes <cli-overview-axes>`, CUDA device
  selection, locking GPU clocks (Volta+), changing output formats, and more.
* :ref:`Throughput calculations <throughput-measurements>`: Compute
  and report:

  * Item throughput (elements/second)
  * Global memory bandwidth usage (bytes/second and per-device %-of-peak-bw)

* Multiple output formats: Currently supports markdown (default) and CSV output.
* :ref:`Manual timer mode <explicit-timer-mode>`:
  (optional) Explicitly start/stop timing in a benchmark implementation.
* Multiple measurement types:

  * Cold Measurements:

    * Each sample runs the benchmark once with a clean device L2 cache.
    * GPU and CPU times are reported.

  * Batch Measurements:

    * Executes the benchmark multiple times back-to-back and records total time.
    * Reports the average execution time (total time / number of executions).

  * :ref:`CPU-only Measurements <cpu-only-benchmarks>`:

    * Measures the host-side execution time of a non-GPU benchmark.
    * Not suitable for microbenchmarking.

Check out `GPU Mode talk #56 <https://www.youtube.com/watch?v=CtrqBmYtSEki>`_ for an overview
of the challenges inherent to CUDA kernel benchmarking and how NVBench solves them for you!

-------

.. toctree::
   :maxdepth: 2
   :hidden:

   cpp_benchmarks
   py_benchmarks
   cli_overview
   cpp_api
   python_api
