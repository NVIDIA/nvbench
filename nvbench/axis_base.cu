#include "axis_base.cuh"

namespace nvbench
{

axis_base::~axis_base() = default;

std::unique_ptr<axis_base> axis_base::clone() const
{
  return this->do_clone();
}

} // namespace nvbench
