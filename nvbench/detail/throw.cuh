#pragma once

#include <fmt/format.h>
#include <stdexcept>

#define NVBENCH_THROW(exception_type, format_str, ...)                         \
  throw exception_type(fmt::format("{}:{}: {}",                                \
                                   __FILE__,                                   \
                                   __LINE__,                                   \
                                   fmt::format(format_str, __VA_ARGS__)))

#define NVBENCH_THROW_IF(condition, exception_type, format_str, ...)           \
  do                                                                           \
  {                                                                            \
    if (condition)                                                             \
    {                                                                          \
      NVBENCH_THROW(exception_type, format_str, __VA_ARGS__);                  \
    }                                                                          \
  } while (false)
