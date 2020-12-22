#include <nvbench/int64_axis.cuh>

#include <fmt/format.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace nvbench
{

int64_axis::~int64_axis() = default;

void int64_axis::set_inputs(const std::vector<int64_t> &inputs)
{
  m_inputs = inputs;
  if (!m_is_power_of_two)
  {
    m_values = inputs;
  }
  else
  {
    m_values.resize(inputs.size());

    auto conv = [](int64_t in) -> int64_t {
      if (in < 0 || in >= 64)
      {
        throw std::runtime_error(fmt::format("{}:{}: Input value exceeds valid "
                                             "range "
                                             "for power-of-two mode. "
                                             "Input={} ValidRange=[0, 63]",
                                             __FILE__,
                                             __LINE__,
                                             in));
      }
      return 1ll << in;
    };

    std::transform(inputs.cbegin(), inputs.cend(), m_values.begin(), conv);
  }
}
std::string int64_axis::do_get_user_string(std::size_t i) const
{
  return fmt::to_string(m_inputs[i]);
}
std::string int64_axis::do_get_user_description(std::size_t i) const
{
  return m_is_power_of_two ? fmt::format("2^{} = {}", m_inputs[i], m_values[i])
                           : std::string{};
}

} // namespace nvbench
