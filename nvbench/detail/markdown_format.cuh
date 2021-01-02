#pragma once

namespace nvbench
{

namespace detail
{

struct markdown_format
{
  // Hacked in to just print a basic summary table to stdout. There's lots of
  // room for improvement here.
  void print();
};

} // namespace detail
} // namespace nvbench
