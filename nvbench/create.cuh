#pragma once

#include <nvbench/benchmark.cuh>
#include <nvbench/benchmark_manager.cuh>
#include <nvbench/callable.cuh>
#include <nvbench/type_list.cuh>

#include <memory>

#define NVBENCH_TYPE_AXES(...) nvbench::type_list<__VA_ARGS__>

#define NVBENCH_BENCH(KernelGenerator)                                         \
  NVBENCH_DEFINE_UNIQUE_CALLABLE(KernelGenerator);                             \
  nvbench::benchmark_base &NVBENCH_UNIQUE_IDENTIFIER(benchmark) =              \
    nvbench::benchmark_manager::get()                                          \
      .add(std::make_unique<                                                   \
           nvbench::benchmark<NVBENCH_UNIQUE_IDENTIFIER(KernelGenerator)>>())  \
      .set_name(#KernelGenerator)

#define NVBENCH_BENCH_TYPES(KernelGenerator, TypeAxes)                         \
  NVBENCH_DEFINE_UNIQUE_CALLABLE_TEMPLATE(KernelGenerator);                    \
  nvbench::benchmark_base &NVBENCH_UNIQUE_IDENTIFIER(benchmark) =              \
    nvbench::benchmark_manager::get()                                          \
      .add(std::make_unique<                                                   \
           nvbench::benchmark<NVBENCH_UNIQUE_IDENTIFIER(KernelGenerator),      \
                              TypeAxes>>())                                    \
      .set_name(#KernelGenerator)
