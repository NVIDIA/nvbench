#include <nvbench/type_strings.cuh>

#include <cstdlib>
#include <memory>
#include <string>

#if defined(__GNUC__) || defined(__clang__)
#define NVBENCH_CXXABI_DEMANGLE
#endif

#ifdef NVBENCH_CXXABI_DEMANGLE
#include <cxxabi.h>
#endif

namespace nvbench::detail
{

std::string nvbench::detail::demangle(const std::string &str)
{
#ifdef NVBENCH_CXXABI_DEMANGLE
  std::unique_ptr<char, std::free> demangled =
    abi::__cxx_demangle(str.c_str(), nullptr, nullptr, nullptr);
  return std::string(demangled.get());
#else
  return str;
#endif
};

} // namespace nvbench::detail
