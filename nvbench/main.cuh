#pragma once

#include <nvbench/benchmark_base.cuh>
#include <nvbench/benchmark_manager.cuh>
#include <nvbench/detail/markdown_format.cuh>

#define NVBENCH_MAIN                                                           \
  int main()                                                                   \
  {                                                                            \
    NVBENCH_MAIN_BODY;                                                         \
    return 0;                                                                  \
  }

#define NVBENCH_MAIN_BODY                                                      \
  do                                                                           \
  {                                                                            \
    auto &mgr = nvbench::benchmark_manager::get();                             \
    for (auto &bench_ptr : mgr.get_benchmarks())                               \
    {                                                                          \
      bench_ptr->run();                                                        \
    }                                                                          \
    nvbench::detail::markdown_format printer;                                  \
    printer.print();                                                           \
  } while (false)
