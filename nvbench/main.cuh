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

#include <cstdlib>
#include <iostream>

// Advanced users can rebuild NVBench's `main` function using the macros in this file, or replace
// them with customized implementations. The default implementation is provided below.

#ifndef NVBENCH_MAIN
#define NVBENCH_MAIN                                                                               \
  int main(int argc, char const *const *argv) { return nvbench::detail::nvbench_main(argc, argv); }
#endif

#ifndef NVBENCH_MAIN_BODY
#define NVBENCH_MAIN_BODY(argc, argv) nvbench::detail::nvbench_main_body(argc, argv)
#endif

#ifndef NVBENCH_MAIN_INITIALIZE
#define NVBENCH_MAIN_INITIALIZE() nvbench::detail::nvbench_main_initialize()
#endif

#ifndef NVBENCH_MAIN_PARSE
#define NVBENCH_MAIN_PARSE(argc, argv)                                                             \
  nvbench::option_parser parser;                                                                   \
  nvbench::detail::nvbench_main_parse(argc, argv, parser)
#endif

#ifndef NVBENCH_MAIN_PRINT_PREAMBLE
#define NVBENCH_MAIN_PRINT_PREAMBLE() nvbench::detail::nvbench_main_print_preamble(parser)
#endif

#ifndef NVBENCH_MAIN_RUN_BENCHMARKS
#define NVBENCH_MAIN_RUN_BENCHMARKS() nvbench::detail::nvbench_main_run_benchmarks(parser)
#endif

#ifndef NVBENCH_MAIN_PRINT_EPILOGUE
#define NVBENCH_MAIN_PRINT_EPILOGUE() nvbench::detail::nvbench_main_print_epilogue(parser)
#endif

#ifndef NVBENCH_MAIN_PRINT_RESULTS
#define NVBENCH_MAIN_PRINT_RESULTS() nvbench::detail::nvbench_main_print_results(parser)
#endif

#ifndef NVBENCH_MAIN_FINALIZE
#define NVBENCH_MAIN_FINALIZE() nvbench::detail::nvbench_main_finalize()
#endif

#ifndef NVBENCH_MAIN_CATCH_EXCEPTIONS
#define NVBENCH_MAIN_CATCH_EXCEPTIONS                                                              \
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
#endif

namespace nvbench::detail
{

inline void set_env(const char *name, const char *value)
{
#ifdef _MSC_VER
  _putenv_s(name, value);
#else
  setenv(name, value, 1);
#endif
}

inline void nvbench_main_initialize()
{
  // See NVIDIA/NVBench#136 for CUDA_MODULE_LOADING
#ifdef _MSC_VER
  _putenv_s("CUDA_MODULE_LOADING", "EAGER");
#else
  setenv("CUDA_MODULE_LOADING", "EAGER", 1);
#endif

  // Initialize CUDA driver API if needed:
#ifdef NVBENCH_HAS_CUPTI
  NVBENCH_DRIVER_API_CALL(cuInit(0));
#endif

  // Initialize the benchmarks *after* setting up the CUDA environment:
  nvbench::benchmark_manager::get().initialize();
}

inline void nvbench_main_parse(int argc, char const *const *argv, option_parser &parser)
{
  parser.parse(argc, argv);
}

inline void nvbench_main_print_preamble(option_parser &parser)
{
  auto &printer = parser.get_printer();

  printer.print_device_info();
  printer.print_log_preamble();
}

inline void nvbench_main_run_benchmarks(option_parser &parser)
{
  auto &printer    = parser.get_printer();
  auto &benchmarks = parser.get_benchmarks();

  std::size_t total_states = 0;
  for (auto &bench_ptr : benchmarks)
  {
    total_states += bench_ptr->get_config_count();
  }

  printer.set_completed_state_count(0);
  printer.set_total_state_count(total_states);

  for (auto &bench_ptr : benchmarks)
  {
    bench_ptr->set_printer(printer);
    bench_ptr->run();
    bench_ptr->clear_printer();
  }
}

inline void nvbench_main_print_epilogue(option_parser &parser)
{
  auto &printer = parser.get_printer();
  printer.print_log_epilogue();
}

inline void nvbench_main_print_results(option_parser &parser)
{
  auto &printer    = parser.get_printer();
  auto &benchmarks = parser.get_benchmarks();
  printer.print_benchmark_results(benchmarks);
}

inline void nvbench_main_finalize() { NVBENCH_CUDA_CALL(cudaDeviceReset()); }

inline int nvbench_main_body(int argc, char const *const *argv)
{
  NVBENCH_MAIN_INITIALIZE();

  {
    NVBENCH_MAIN_PARSE(argc, argv);

    NVBENCH_MAIN_PRINT_PREAMBLE();
    NVBENCH_MAIN_RUN_BENCHMARKS();
    NVBENCH_MAIN_PRINT_EPILOGUE();

    NVBENCH_MAIN_PRINT_RESULTS();
  } // Tear down parser before finalization

  NVBENCH_MAIN_FINALIZE();

  return 0;
}

inline int nvbench_main(int argc, char const *const *argv)
try
{
  NVBENCH_MAIN_BODY(argc, argv);
  return 0;
}
NVBENCH_MAIN_CATCH_EXCEPTIONS

} // namespace nvbench::detail
