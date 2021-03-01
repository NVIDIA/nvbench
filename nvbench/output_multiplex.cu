#include <nvbench/output_multiplex.cuh>

#include <iostream>

namespace nvbench
{

output_multiplex::output_multiplex()
    : output_format(std::cerr) // Nothing should write to this.
{}

void output_multiplex::do_print_device_info()
{
  for (auto &format_ptr : m_formats)
  {
    format_ptr->print_device_info();
  }
}

void output_multiplex::do_print_log_preamble()
{
  for (auto &format_ptr : m_formats)
  {
    format_ptr->print_log_preamble();
  }
}

void output_multiplex::do_print_log_epilogue()
{
  for (auto &format_ptr : m_formats)
  {
    format_ptr->print_log_epilogue();
  }
}

void output_multiplex::do_print_benchmark_list(const benchmark_vector &benches)
{
  for (auto &format_ptr : m_formats)
  {
    format_ptr->print_benchmark_list(benches);
  }
}

void output_multiplex::do_print_benchmark_results(
  const benchmark_vector &benches)
{
  for (auto &format_ptr : m_formats)
  {
    format_ptr->print_benchmark_results(benches);
  }
}

} // namespace nvbench
