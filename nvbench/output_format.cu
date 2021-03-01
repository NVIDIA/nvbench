#include <nvbench/output_format.cuh>

#include <ostream>

namespace nvbench
{

output_format::output_format(std::ostream &ostream)
    : m_ostream{ostream}
{}

// Defined here to keep <ostream> out of the header
output_format::~output_format() = default;

} // namespace nvbench
