/*
 *  Copyright 2026 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 with the LLVM exception
 *  (the "License"); you may not use this file except in compliance with
 *  the License. You may obtain a copy of the License at
 *
 *      http://llvm.org/foundation/relicensing/LICENSE.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <charconv>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

int g_iterations = 1;

void parse_custom_args(std::vector<std::string> &args)
{
  const auto option = std::find(args.begin(), args.end(), "--iterations");
  if (option == args.end())
  {
    return;
  }

  const auto value = std::next(option);
  if (value == args.end())
  {
    throw std::invalid_argument("--iterations requires a positive integer");
  }

  int iterations    = 0;
  const auto result = std::from_chars(value->data(), value->data() + value->size(), iterations);
  if (result.ec != std::errc{} || result.ptr != value->data() + value->size() || iterations <= 0)
  {
    throw std::invalid_argument("--iterations requires a positive integer");
  }

  g_iterations = iterations;
  args.erase(option, std::next(value));
}

} // namespace

// Remove application-specific arguments before NVBench parses the remaining CLI.
#undef NVBENCH_MAIN_CUSTOM_ARGS_HANDLER
#define NVBENCH_MAIN_CUSTOM_ARGS_HANDLER(args) parse_custom_args(args)

void custom_args_bench(nvbench::state &state)
{
  state.exec(nvbench::exec_tag::no_gpu, [](nvbench::launch &) {
    volatile int sink = 0;
    for (int i = 0; i < g_iterations; ++i)
    {
      sink += i;
    }
    (void)sink;
  });
}
NVBENCH_BENCH(custom_args_bench).set_is_cpu_only(true);

// Example invocation:
//   nvbench.example.custom_args --iterations 4 --benchmark custom_args_bench
NVBENCH_MAIN
