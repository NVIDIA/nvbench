#pragma once

#include <fmt/format.h>

#define ASSERT(cond)                                                            \
  do                                                                            \
  {                                                                             \
    if (cond)                                                                   \
    {}                                                                          \
    else                                                                        \
    {                                                                           \
      fmt::print("{}:{}: Assertion failed ({}).\n", __FILE__, __LINE__, #cond); \
      exit(EXIT_FAILURE);                                                       \
    }                                                                           \
  } while (false)

#define ASSERT_MSG(cond, msg)                                                  \
  do                                                                           \
  {                                                                            \
    if (cond)                                                                  \
    {}                                                                         \
    else                                                                       \
    {                                                                          \
      fmt::print("{}:{}: Test assertion failed ({}) {}\n",                     \
                 __FILE__,                                                     \
                 __LINE__,                                                     \
                 #cond,                                                        \
                 msg);                                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (false)
