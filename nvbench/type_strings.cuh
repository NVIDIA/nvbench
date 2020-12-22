#pragma once

#include <nvbench/types.cuh>

#include <string>
#include <typeinfo>

namespace nvbench
{

template <typename T>
struct type_strings
{
  // TODO demangle on GCC/Clang

  // The string used to identify the type in shorthand (e.g. output tables and
  // CLI options):
  static std::string input_string() { return typeid(T).name(); }

  // A more descriptive identifier for the type, if input_string is not a common
  // identifier. May be blank if `input_string` is obvious.
  static std::string description() { return {}; }
};

} // namespace nvbench

#define NVBENCH_DECLARE_TYPE_STRINGS(Type, InputString, Description)           \
  namespace nvbench                                                            \
  {                                                                            \
  template <>                                                                  \
  struct type_strings<Type>                                                    \
  {                                                                            \
    static std::string input_string() { return {InputString}; }                \
    static std::string description() { return {Description}; }                 \
  };                                                                           \
  }

NVBENCH_DECLARE_TYPE_STRINGS(nvbench::int8_t, "I8", "int8_t");
NVBENCH_DECLARE_TYPE_STRINGS(nvbench::int16_t, "I16", "int16_t");
NVBENCH_DECLARE_TYPE_STRINGS(nvbench::int32_t, "I32", "int32_t");
NVBENCH_DECLARE_TYPE_STRINGS(nvbench::int64_t, "I64", "int64_t");
NVBENCH_DECLARE_TYPE_STRINGS(nvbench::uint8_t, "U8", "uint8_t");
NVBENCH_DECLARE_TYPE_STRINGS(nvbench::uint16_t, "U16", "uint16_t");
NVBENCH_DECLARE_TYPE_STRINGS(nvbench::uint32_t, "U32", "uint32_t");
NVBENCH_DECLARE_TYPE_STRINGS(nvbench::uint64_t, "U64", "uint64_t");
NVBENCH_DECLARE_TYPE_STRINGS(nvbench::float32_t, "F32", "float");
NVBENCH_DECLARE_TYPE_STRINGS(nvbench::float64_t, "F64", "double");
