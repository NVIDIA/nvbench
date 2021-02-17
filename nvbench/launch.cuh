#pragma once

#include <nvbench/cuda_stream.cuh>

namespace nvbench
{

struct launch
{
  // move-only
  launch()               = default;
  launch(const launch &) = delete;
  launch(launch &&)      = default;
  launch &operator=(const launch &) = delete;
  launch &operator=(launch &&) = default;

  __forceinline__ const nvbench::cuda_stream &get_stream() const
  {
    return m_stream;
  };

private:
  nvbench::cuda_stream m_stream;
};

} // namespace nvbench
