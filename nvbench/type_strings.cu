#include <nvbench/type_strings.cuh>

#include <string>

#if defined(__GNUC__) || defined(__clang__)
#define NVBENCH_CXXABI_DEMANGLE
#endif

#ifdef NVBENCH_CXXABI_DEMANGLE
#include <cxxabi.h>

#include <cstdlib>
#include <memory>

namespace
{
struct free_wrapper
{
  void operator()(void* ptr) { std::free(ptr); }
};
} // end namespace

#endif // NVBENCH_CXXABI_DEMANGLE

namespace nvbench::detail
{

std::string demangle(const std::string &str)
{
#ifdef NVBENCH_CXXABI_DEMANGLE
  std::unique_ptr<char, free_wrapper> demangled{
    abi::__cxa_demangle(str.c_str(), nullptr, nullptr, nullptr)};
  return std::string(demangled.get());
#else
  return str;
#endif
};

} // namespace nvbench::detail
