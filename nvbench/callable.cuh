#pragma once

#include <nvbench/type_list.cuh>

#include <type_traits>

namespace nvbench
{
struct state;
}

// Define a simple callable wrapper around a function. This allows the function
// to be used as a class template parameter. Intended for use with kernel
// generators and `NVBENCH_CREATE` macros.
#define NVBENCH_DEFINE_UNIQUE_CALLABLE(function)                               \
  NVBENCH_DEFINE_CALLABLE(function, NVBENCH_UNIQUE_IDENTIFIER(function))

#define NVBENCH_DEFINE_CALLABLE(function, callable_name)                       \
  struct callable_name                                                         \
  {                                                                            \
    void operator()(nvbench::state &state, nvbench::type_list<>)               \
    {                                                                          \
      function(state);                                                         \
    }                                                                          \
  }

#define NVBENCH_DEFINE_UNIQUE_CALLABLE_TEMPLATE(function)                      \
  NVBENCH_DEFINE_CALLABLE_TEMPLATE(function,                                   \
                                   NVBENCH_UNIQUE_IDENTIFIER(function))

#define NVBENCH_DEFINE_CALLABLE_TEMPLATE(function, callable_name)              \
  struct callable_name                                                         \
  {                                                                            \
    template <typename... Ts>                                                  \
    void operator()(nvbench::state &state, nvbench::type_list<Ts...>)          \
    {                                                                          \
      function(state, nvbench::type_list<Ts...>{});                            \
    }                                                                          \
  }

#define NVBENCH_UNIQUE_IDENTIFIER(prefix)                                      \
  NVBENCH_UNIQUE_IDENTIFIER_IMPL1(prefix, __LINE__)
#define NVBENCH_UNIQUE_IDENTIFIER_IMPL1(prefix, unique_id)                     \
  NVBENCH_UNIQUE_IDENTIFIER_IMPL2(prefix, unique_id)
#define NVBENCH_UNIQUE_IDENTIFIER_IMPL2(prefix, unique_id)                     \
  prefix##_line_##unique_id
