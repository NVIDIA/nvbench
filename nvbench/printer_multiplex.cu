/*
 *  Copyright 2020 NVIDIA Corporation
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

#include <nvbench/printer_multiplex.cuh>

#include <iostream>

namespace nvbench
{

printer_multiplex::printer_multiplex()
    : printer_base(std::cerr) // Nothing should write to this.
{}

void printer_multiplex::do_print_device_info()
{
  for (auto &format_ptr : m_printers)
  {
    format_ptr->print_device_info();
  }
}

void printer_multiplex::do_print_log_preamble()
{
  for (auto &format_ptr : m_printers)
  {
    format_ptr->print_log_preamble();
  }
}

void printer_multiplex::do_print_log_epilogue()
{
  for (auto &format_ptr : m_printers)
  {
    format_ptr->print_log_epilogue();
  }
}

void printer_multiplex::do_log(nvbench::log_level level, const std::string &str)
{
  for (auto &format_ptr : m_printers)
  {
    format_ptr->log(level, str);
  }
}

void printer_multiplex::do_log_run_state(const nvbench::state &exec_state)
{
  for (auto &format_ptr : m_printers)
  {
    format_ptr->log_run_state(exec_state);
  }
}

void printer_multiplex::do_print_benchmark_list(const benchmark_vector &benches)
{
  for (auto &format_ptr : m_printers)
  {
    format_ptr->print_benchmark_list(benches);
  }
}

void printer_multiplex::do_print_benchmark_results(
  const benchmark_vector &benches)
{
  for (auto &format_ptr : m_printers)
  {
    format_ptr->print_benchmark_results(benches);
  }
}

} // namespace nvbench
