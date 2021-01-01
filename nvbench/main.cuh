#pragma once

#include <nvbench/benchmark_base.cuh>
#include <nvbench/benchmark_manager.cuh>

#define NVBENCH_MAIN                                                           \
  int main() { BENCHMARK_MAIN_BODY; }

#define NVBENCH_MAIN_BODY                                                      \
  do                                                                           \
  {                                                                            \
    auto &mgr = nvbench::benchmark_manager::get();                             \
    for (auto &bench_ptr : mgr.get_benchmarks())                               \
    {                                                                          \
      bench_ptr->run();                                                        \
    }                                                                          \
  } while (false)
