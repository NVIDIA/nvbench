#include <nvbench/type_list.cuh>

#include <cstdint>
#include <type_traits>

// Unique, numbered types for testing type_list functionality.
using T0 = std::integral_constant<std::size_t, 0>;
using T1 = std::integral_constant<std::size_t, 1>;
using T2 = std::integral_constant<std::size_t, 2>;
using T3 = std::integral_constant<std::size_t, 3>;
using T4 = std::integral_constant<std::size_t, 4>;
using T5 = std::integral_constant<std::size_t, 5>;
using T6 = std::integral_constant<std::size_t, 6>;
using T7 = std::integral_constant<std::size_t, 7>;

struct test_get
{
  using TL = nvbench::type_list<T0, T1, T2, T3, T4, T5>;
  static_assert(std::is_same_v<T0, nvbench::tl::get<0, TL>>);
  static_assert(std::is_same_v<T1, nvbench::tl::get<1, TL>>);
  static_assert(std::is_same_v<T2, nvbench::tl::get<2, TL>>);
  static_assert(std::is_same_v<T3, nvbench::tl::get<3, TL>>);
  static_assert(std::is_same_v<T4, nvbench::tl::get<4, TL>>);
  static_assert(std::is_same_v<T5, nvbench::tl::get<5, TL>>);
};

struct test_concat
{
  using TLEmpty = nvbench::type_list<>;
  using TL012   = nvbench::type_list<T0, T1, T2>;
  using TL765   = nvbench::type_list<T7, T6, T5>;

  struct empty_tests
  {
    static_assert(
      std::is_same_v<nvbench::tl::concat<TLEmpty, TLEmpty>, TLEmpty>);
    static_assert(std::is_same_v<nvbench::tl::concat<TLEmpty, TL012>, TL012>);
    static_assert(std::is_same_v<nvbench::tl::concat<TL012, TLEmpty>, TL012>);
  };

  static_assert(std::is_same_v<nvbench::tl::concat<TL012, TL765>,
                               nvbench::type_list<T0, T1, T2, T7, T6, T5>>);
};

struct test_prepend_each
{
  using T   = void;
  using T01 = nvbench::type_list<T0, T1>;
  using T23 = nvbench::type_list<T2, T3>;
  using TLs = nvbench::type_list<T01, T23>;

  using Expected = nvbench::type_list<nvbench::type_list<T, T0, T1>,
                                      nvbench::type_list<T, T2, T3>>;
  static_assert(std::is_same_v<nvbench::tl::prepend_each<T, TLs>, Expected>);
};

struct test_cartesian_product
{
  using U0       = T2;
  using U1       = T3;
  using U2       = T4;
  using V0       = T5;
  using V1       = T6;
  using T01      = nvbench::type_list<T0, T1>;
  using U012     = nvbench::type_list<U0, U1, U2>;
  using V01      = nvbench::type_list<V0, V1>;
  using TLs      = nvbench::type_list<T01, U012, V01>;
  using CartProd = nvbench::type_list<nvbench::type_list<T0, U0, V0>,
                                      nvbench::type_list<T0, U0, V1>,
                                      nvbench::type_list<T0, U1, V0>,
                                      nvbench::type_list<T0, U1, V1>,
                                      nvbench::type_list<T0, U2, V0>,
                                      nvbench::type_list<T0, U2, V1>,
                                      nvbench::type_list<T1, U0, V0>,
                                      nvbench::type_list<T1, U0, V1>,
                                      nvbench::type_list<T1, U1, V0>,
                                      nvbench::type_list<T1, U1, V1>,
                                      nvbench::type_list<T1, U2, V0>,
                                      nvbench::type_list<T1, U2, V1>>;
  static_assert(std::is_same_v<nvbench::tl::cartesian_product<TLs>, CartProd>);
};

// This test only has static asserts.
int main() {}
