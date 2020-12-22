#pragma once

#include <type_traits>

#define NVBENCH_DECLARE_FLAGS(T)                                               \
  inline T operator|(T v1, T v2)                                               \
  {                                                                            \
    using UT = std::underlying_type_t<T>;                                      \
    return static_cast<T>(static_cast<UT>(v1) | static_cast<UT>(v2));          \
  }                                                                            \
  inline T operator&(T v1, T v2)                                               \
  {                                                                            \
    using UT = std::underlying_type_t<T>;                                      \
    return static_cast<T>(static_cast<UT>(v1) & static_cast<UT>(v2));          \
  }                                                                            \
  inline T operator^(T v1, T v2)                                               \
  {                                                                            \
    using UT = std::underlying_type_t<T>;                                      \
    return static_cast<T>(static_cast<UT>(v1) ^ static_cast<UT>(v2));          \
  }                                                                            \
  inline T operator~(T v1)                                                     \
  {                                                                            \
    using UT = std::underlying_type_t<T>;                                      \
    return static_cast<T>(~static_cast<UT>(v1));                               \
  }
