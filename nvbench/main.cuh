/*
 *  Copyright 2021 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 with the LLVM exception
 *  (the "License"); you may not use this file except in compliance with
 *  the License.
 *
 *  You may obtain a copy of the License at
 *
 *      http://llvm.org/foundation/relicensing/LICENSE.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <nvbench/benchmark_base.cuh>
#include <nvbench/benchmark_manager.cuh>
#include <nvbench/config.cuh>
#include <nvbench/cuda_call.cuh>
#include <nvbench/option_parser.cuh>
#include <nvbench/printer_base.cuh>

#include <iostream>

#define NVBENCH_MAIN                                                                               \
  int main(int argc, char const *const *argv)                                                      \
  try                                                                                              \
  {                                                                                                \
    NVBENCH_MAIN_BODY(argc, argv);                                                                 \
    NVBENCH_CUDA_CALL(cudaDeviceReset());                                                          \
    return 0;                                                                                      \
  }                                                                                                \
  catch (std::exception & e)                                                                       \
  {                                                                                                \
    std::cerr << "\nNVBench encountered an error:\n\n" << e.what() << "\n";                        \
    return 1;                                                                                      \
  }                                                                                                \
  catch (...)                                                                                      \
  {                                                                                                \
    std::cerr << "\nNVBench encountered an unknown error.\n";                                      \
    return 1;                                                                                      \
  }

#ifdef NVBENCH_HAS_CUPTI
#define NVBENCH_INITIALIZE_DRIVER_API NVBENCH_DRIVER_API_CALL(cuInit(0))
#else
// clang-format off
#define NVBENCH_INITIALIZE_DRIVER_API do {} while (false)
// clang-format on
#endif

#define NVBENCH_MAIN_PARSE(argc, argv)                                                             \
  nvbench::option_parser parser;                                                                   \
  parser.parse(argc, argv)

#define NVBENCH_MAIN_BODY(argc, argv)                                                              \
  do                                                                                               \
  {                                                                                                \
    NVBENCH_INITIALIZE_DRIVER_API;                                                                 \
    NVBENCH_MAIN_PARSE(argc, argv);                                                                \
    auto &printer = parser.get_printer();                                                          \
                                                                                                   \
    printer.print_device_info();                                                                   \
    printer.print_log_preamble();                                                                  \
    auto &benchmarks = parser.get_benchmarks();                                                    \
                                                                                                   \
    std::size_t total_states = 0;                                                                  \
    for (auto &bench_ptr : benchmarks)                                                             \
    {                                                                                              \
      total_states += bench_ptr->get_config_count();                                               \
    }                                                                                              \
    printer.set_total_state_count(total_states);                                                   \
                                                                                                   \
    printer.set_completed_state_count(0);                                                          \
    for (auto &bench_ptr : benchmarks)                                                             \
    {                                                                                              \
      bench_ptr->set_printer(printer);                                                             \
      bench_ptr->run();                                                                            \
      bench_ptr->clear_printer();                                                                  \
    }                                                                                              \
    printer.print_log_epilogue();                                                                  \
    printer.print_benchmark_results(benchmarks);                                                   \
  } while (false)
