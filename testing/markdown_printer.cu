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

#include <nvbench/markdown_printer.cuh>

#include <sstream>
#include <string>
#include <vector>

#include "test_asserts.cuh"

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

int main() { test_argv_fence_grows_for_backticks(); }
