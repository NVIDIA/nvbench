#pragma once

#include <nvbench/benchmark_base.cuh>
#include <nvbench/benchmark_manager.cuh>
#include <nvbench/detail/markdown_format.cuh>
#include <nvbench/option_parser.cuh>

#define NVBENCH_MAIN                                                           \
  int main(int argc, char const *const *argv)                                  \
  {                                                                            \
    NVBENCH_MAIN_BODY(argc, argv);                                             \
    return 0;                                                                  \
  }

#define NVBENCH_MAIN_BODY(argc, argv)                                          \
  do                                                                           \
  {                                                                            \
    nvbench::detail::markdown_format printer;                                  \
    printer.print_device_info();                                               \
    printer.print_log_preamble();                                              \
    nvbench::option_parser parser;                                             \
    parser.parse(argc, argv);                                                  \
    auto &benchmarks = parser.get_benchmarks();                                \
    for (auto &bench_ptr : benchmarks)                                         \
    {                                                                          \
      bench_ptr->run();                                                        \
    }                                                                          \
    printer.print_log_epilogue();                                              \
    printer.print_benchmark_summaries(benchmarks);                             \
  } while (false)
