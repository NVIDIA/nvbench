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
  NVBENCH_DEFINE_CALLABLE(function, NVBENCH_UNIQUE_CALLABLE_NAME(function))

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
                                   NVBENCH_UNIQUE_CALLABLE_NAME(function))

#define NVBENCH_DEFINE_CALLABLE_TEMPLATE(function, callable_name)              \
  struct callable_name                                                         \
  {                                                                            \
    template <typename... Ts>                                                  \
    void operator()(nvbench::state &state, nvbench::type_list<Ts...>)          \
    {                                                                          \
      function(state, nvbench::type_list<Ts...>{});                            \
    }                                                                          \
  }

#define NVBENCH_UNIQUE_CALLABLE_NAME(function)                                 \
  NVBENCH_UNIQUE_CALLABLE_NAME_IMPL(function, __LINE__)

#define NVBENCH_UNIQUE_CALLABLE_NAME_IMPL(function, unique_id)                 \
  callable_name##_callable_##unique_id
