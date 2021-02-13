#pragma once

#include <nvbench/benchmark_base.cuh>
#include <nvbench/benchmark_manager.cuh>
#include <nvbench/cuda_call.cuh>
#include <nvbench/detail/markdown_format.cuh>
#include <nvbench/option_parser.cuh>

#include <iostream>

#define NVBENCH_MAIN                                                           \
  int main(int argc, char const *const *argv)                                  \
  try                                                                          \
  {                                                                            \
    NVBENCH_MAIN_BODY(argc, argv);                                             \
    NVBENCH_CUDA_CALL(cudaDeviceReset());                                      \
    return 0;                                                                  \
  }                                                                            \
  catch (std::exception & e)                                                   \
  {                                                                            \
    std::cerr << "\nNVBench encountered an error:\n\n" << e.what() << "\n";    \
    return 1;                                                                  \
  }                                                                            \
  catch (...)                                                                  \
  {                                                                            \
    std::cerr << "\nNVBench encountered an unknown error.\n";                  \
    return 1;                                                                  \
  }

#define NVBENCH_MAIN_BODY(argc, argv)                                          \
  do                                                                           \
  {                                                                            \
    nvbench::option_parser parser;                                             \
    parser.parse(argc, argv);                                                  \
                                                                               \
    nvbench::detail::markdown_format printer;                                  \
    printer.print_device_info();                                               \
    printer.print_log_preamble();                                              \
    auto &benchmarks = parser.get_benchmarks();                                \
    for (auto &bench_ptr : benchmarks)                                         \
    {                                                                          \
      bench_ptr->run();                                                        \
    }                                                                          \
    printer.print_log_epilogue();                                              \
    printer.print_benchmark_results(benchmarks);                               \
  } while (false)
