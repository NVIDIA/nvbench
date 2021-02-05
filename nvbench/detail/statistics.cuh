#pragma once

#include <functional>
#include <numeric>
#include <vector>

namespace nvbench::detail
{

inline nvbench::float64_t
compute_stdev(const std::vector<nvbench::float64_t> &data,
              nvbench::float64_t mean)
{
  const auto num  = static_cast<nvbench::float64_t>(data.size());
  const auto sig2 = std::transform_reduce(data.cbegin(),
                                          data.cend(),
                                          0.,
                                          std::plus<>{},
                                          [mean](nvbench::float64_t val) {
                                            val -= mean;
                                            val *= val;
                                            return val;
                                          }) /
                    num;
  return std::sqrt(sig2);
}

} // namespace nvbench::detail
