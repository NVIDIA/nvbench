#pragma once

#include <fmt/format.h>

#include <cstdio>
#include <stdexcept>

#define ASSERT(cond)                                                            \
  do                                                                            \
  {                                                                             \
    if (cond)                                                                   \
    {}                                                                          \
    else                                                                        \
    {                                                                           \
      fmt::print("{}:{}: Assertion failed ({}).\n", __FILE__, __LINE__, #cond); \
      std::fflush(stdout);                                                      \
      throw std::runtime_error("Unit test failure.");                           \
    }                                                                           \
  } while (false)

#define ASSERT_MSG(cond, fmtstr, ...)                                          \
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
                 fmt::format(fmtstr, __VA_ARGS__));                            \
      std::fflush(stdout);                                                      \
      throw std::runtime_error("Unit test failure.");                          \
    }                                                                          \
  } while (false)

#define ASSERT_THROWS_ANY(expr)                                                \
  do                                                                           \
  {                                                                            \
    bool threw = false;                                                        \
    try                                                                        \
    {                                                                          \
      expr;                                                                    \
    }                                                                          \
    catch (...)                                                                \
    {                                                                          \
      threw = true;                                                            \
    }                                                                          \
    if (!threw)                                                                \
    {                                                                          \
      fmt::print("{}:{}: Expression expected exception: '{}'.",                \
                 __FILE__,                                                     \
                 __LINE__,                                                     \
                 #expr);                                                       \
      std::fflush(stdout);                                                      \
      throw std::runtime_error("Unit test failure.");                          \
    }                                                                          \
  } while (false)
