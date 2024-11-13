# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Passes all args directly to execute_process while setting up the following
# results variables and propogating them to the caller's scope:
#
# - nvbench_process_exit_code
# - nvbench_process_stdout
# - nvbench_process_stderr
#
# If the command is not successful (e.g. the last command does not return zero),
# a non-fatal warning is printed.
function(nvbench_execute_non_fatal_process)
  execute_process(${ARGN}
    RESULT_VARIABLE nvbench_process_exit_code
    OUTPUT_VARIABLE nvbench_process_stdout
    ERROR_VARIABLE nvbench_process_stderr
  )

  if (NOT nvbench_process_exit_code EQUAL 0)
    message(WARNING
      "execute_process failed with non-zero exit code: ${nvbench_process_exit_code}\n"
      "${ARGN}\n"
      "stdout:\n${nvbench_process_stdout}\n"
      "stderr:\n${nvbench_process_stderr}\n"
    )
  endif()

  set(nvbench_process_exit_code "${nvbench_process_exit_code}" PARENT_SCOPE)
  set(nvbench_process_stdout "${nvbench_process_stdout}" PARENT_SCOPE)
  set(nvbench_process_stderr "${nvbench_process_stderr}" PARENT_SCOPE)
endfunction()

# Writes CMAKE_CUDA_ARCHITECTURES to out_var, but using escaped semicolons
# as delimiters
function(nvbench_escaped_cuda_arches out_var)
  set(tmp)
  set(first TRUE)
  foreach(arg IN LISTS CMAKE_CUDA_ARCHITECTURES)
    if (NOT first)
      string(APPEND tmp "\;")
    endif()
    string(APPEND tmp "${arg}")
    set(first FALSE)
  endforeach()
  set(${out_var} "${tmp}" PARENT_SCOPE)
endfunction()
