/*
 *  Copyright 2026 NVIDIA Corporation
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

#include <nvbench/benchmark.cuh>
#include <nvbench/callable.cuh>
#include <nvbench/markdown_printer.cuh>

#include <sstream>
#include <string>
#include <vector>

#include "test_asserts.cuh"

void no_op_generator(nvbench::state &) {}
NVBENCH_DEFINE_CALLABLE(no_op_generator, no_op_callable);
using no_op_bench = nvbench::benchmark<no_op_callable>;

void test_argv_fence_grows_for_backticks()
{
  std::ostringstream output;
  nvbench::markdown_printer printer{output};

  printer.log_argv({"benchmark"});
  printer.log_raw_argv({"benchmark", "contains```fence"});
  printer.print_argv();

  const auto markdown = output.str();
  ASSERT(markdown.find("# Command Line\n\n````\n") != std::string::npos);
  ASSERT(markdown.find("contains```fence") != std::string::npos);
  ASSERT(markdown.find("\n````\n\n") != std::string::npos);
}

void test_benchmark_list_includes_description()
{
  std::ostringstream output;
  nvbench::markdown_printer printer{output};
  no_op_bench bench;
  bench.set_name("no_op_generator");
  bench.set_description("Measures the execution time of my kernel.");

  nvbench::printer_base::benchmark_vector benches;
  benches.emplace_back(bench.clone());
  printer.print_benchmark_list(benches);

  const auto markdown = output.str();
  ASSERT(markdown.find("`no_op_generator`") != std::string::npos);
  ASSERT(markdown.find("Measures the execution time of my kernel.") != std::string::npos);
}

int main()
{
  test_argv_fence_grows_for_backticks();
  test_benchmark_list_includes_description();
}
