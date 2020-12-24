#include <nvbench/int64_axis.cuh>

#include <fmt/format.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace nvbench
{

int64_axis::~int64_axis() = default;

void int64_axis::set_inputs(std::vector<int64_t> inputs)
{
  m_inputs = std::move(inputs);
  if (!this->is_power_of_two())
  {
    m_values = m_inputs;
  }
  else
  {
    m_values.resize(m_inputs.size());

    auto conv = [](int64_t in) -> int64_t {
      if (in < 0 || in >= 64)
      {
        throw std::runtime_error(fmt::format("{}:{}: Input value exceeds valid "
                                             "range for power-of-two mode. "
                                             "Input={} ValidRange=[0, 63]",
                                             __FILE__,
                                             __LINE__,
                                             in));
      }
      return 1ll << in;
    };

    std::transform(m_inputs.cbegin(), m_inputs.cend(), m_values.begin(), conv);
  }
}

std::string int64_axis::do_get_input_string(std::size_t i) const
{
  return fmt::to_string(m_inputs[i]);
}

std::string int64_axis::do_get_description(std::size_t i) const
{
  return this->is_power_of_two()
           ? fmt::format("2^{} = {}", m_inputs[i], m_values[i])
           : std::string{};
}

} // namespace nvbench
