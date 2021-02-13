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

template <typename InT,
          typename OutT = nvbench::detail::range_output_t<InT>>
auto range(InT start, InT end, InT stride = InT{1})
{
  using result_t = std::vector<OutT>;
  result_t result;
  for (; start <= end; start += stride)
  {
    result.push_back(static_cast<OutT>(start));
  }
  return result;
}

} // namespace nvbench
