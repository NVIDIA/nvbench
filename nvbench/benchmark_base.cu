#include <nvbench/benchmark_base.cuh>

namespace nvbench
{

benchmark_base::~benchmark_base() = default;

std::unique_ptr<benchmark_base> benchmark_base::clone() const
{
  auto result = this->do_clone();

  // Do not copy states.
  result->m_name = m_name;
  result->m_axes = m_axes;

  return std::move(result);
}


} // namespace nvbench
