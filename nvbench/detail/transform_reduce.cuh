#pragma once

// Many compilers still don't ship transform_reduce with their STLs, so here's
// a naive implementation that will work everywhere. This is never used in a
// critical section, so perf isn't a concern.

namespace nvbench::detail
{

template <typename InIterT,
          typename InitValueT,
          typename ReduceOp,
          typename TransformOp>
InitValueT transform_reduce(InIterT first,
                            InIterT last,
                            InitValueT init,
                            ReduceOp &&reduceOp,
                            TransformOp &&transformOp)
{
  while (first != last)
  {
    init = reduceOp(std::move(init), transformOp(*first++));
  }
  return init;
}

} // namespace nvbench::detail
