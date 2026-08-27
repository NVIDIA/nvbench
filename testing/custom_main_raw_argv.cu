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

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

// Rewrite "--my-custom-arg" into "--profile". The reported command line must keep
// the original argument.
void custom_arg_handler(std::vector<std::string> &args)
{
  auto it = std::find(args.begin(), args.end(), "--my-custom-arg");
  if (it == args.end())
  {
    throw std::runtime_error("Custom argument not found.");
  }
  *it = "--profile";
}

#undef NVBENCH_MAIN_CUSTOM_ARGS_HANDLER
#define NVBENCH_MAIN_CUSTOM_ARGS_HANDLER(args) custom_arg_handler(args)

void verify(nvbench::option_parser &parser)
{
  const auto &raw    = parser.get_raw_args();
  const auto &parsed = parser.get_args();

  if (std::find(raw.begin(), raw.end(), "--my-custom-arg") == raw.end())
  {
    throw std::runtime_error("Raw args lost the original argument.");
  }
  if (std::find(raw.begin(), raw.end(), "--profile") != raw.end())
  {
    throw std::runtime_error("Raw args contain the rewritten argument.");
  }
  if (std::find(parsed.begin(), parsed.end(), "--profile") == parsed.end())
  {
    throw std::runtime_error("Parsed args lost the rewritten argument.");
  }
}

#undef NVBENCH_MAIN_PARSE_CUSTOM_POST
#define NVBENCH_MAIN_PARSE_CUSTOM_POST(parser) verify(parser)

void bench(nvbench::state &state)
{
  state.exec([](nvbench::launch &) {});
}
NVBENCH_BENCH(bench);

NVBENCH_MAIN
