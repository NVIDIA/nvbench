#pragma once

#include <cstdint>
#include <tuple>

namespace nvbench
{

template <typename... Ts>
struct type_list;

namespace tl
{

namespace detail
{

template <typename... Ts>
auto size(nvbench::type_list<Ts...>)
  -> std::integral_constant<std::size_t, sizeof...(Ts)>;

template <std::size_t I, typename... Ts>
auto get(nvbench::type_list<Ts...>)
  -> std::tuple_element_t<I, std::tuple<Ts...>>;

template <typename... Ts, typename... Us>
auto concat(nvbench::type_list<Ts...>, nvbench::type_list<Us...>)
  -> nvbench::type_list<Ts..., Us...>;

//------------------------------------------------------------------------------
template <typename T, typename TLs>
struct prepend_each;

template <typename T>
struct prepend_each<T, nvbench::type_list<>>
{
  using type = nvbench::type_list<>;
};

template <typename T, typename TL, typename... TLTail>
struct prepend_each<T, nvbench::type_list<TL, TLTail...>>
{
  using cur = decltype(detail::concat(nvbench::type_list<T>{}, TL{}));
  using next =
    typename detail::prepend_each<T, nvbench::type_list<TLTail...>>::type;
  using type = decltype(detail::concat(nvbench::type_list<cur>{}, next{}));
};

//------------------------------------------------------------------------------
template <typename TLs>
struct cartesian_product;

template <>
struct cartesian_product<nvbench::type_list<>>
{
  using type = nvbench::type_list<>;
};

template <typename... TLTail>
struct cartesian_product<nvbench::type_list<nvbench::type_list<>, TLTail...>>
{
  using type = nvbench::type_list<>;
};

template <typename T, typename... Ts>
struct cartesian_product<nvbench::type_list<nvbench::type_list<T, Ts...>>>
{
  using cur  = nvbench::type_list<nvbench::type_list<T>>;
  using next = typename detail::cartesian_product<
    nvbench::type_list<nvbench::type_list<Ts...>>>::type;
  using type = decltype(detail::concat(cur{}, next{}));
};

template <typename T, typename... Tail, typename TL, typename... TLTail>
struct cartesian_product<
  nvbench::type_list<nvbench::type_list<T, Tail...>, TL, TLTail...>>
{
  using tail_prod =
    typename detail::cartesian_product<nvbench::type_list<TL, TLTail...>>::type;
  using cur  = typename detail::prepend_each<T, tail_prod>::type;
  using next = typename detail::cartesian_product<
    nvbench::type_list<nvbench::type_list<Tail...>, TL, TLTail...>>::type;
  using type = decltype(detail::concat(cur{}, next{}));
};

} // namespace detail
} // namespace tl
} // namespace nvbench
