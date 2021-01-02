#pragma once

#include <nvbench/types.cuh>

#include <type_traits>
#include <vector>

namespace nvbench
{

namespace detail
{
template <typename T>
using range_output_t = std::conditional_t<std::is_floating_point_v<T>,
                                          nvbench::float64_t,
                                          nvbench::int64_t>;
}

template <typename T>
auto range(T start, T end, T stride = T{1})
{
  using output_t = detail::range_output_t<T>;
  using result_t = std::vector<output_t>;
  result_t result;
  for (; start <= end; start += stride)
  {
    result.push_back(static_cast<output_t>(start));
  }
  return result;
}

} // namespace nvbench
