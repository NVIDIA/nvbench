#include <nvbench/printer_base.cuh>

#include <ostream>

namespace nvbench
{

printer_base::printer_base(std::ostream &ostream)
    : m_ostream{ostream}
{}

// Defined here to keep <ostream> out of the header
printer_base::~printer_base() = default;

} // namespace nvbench
