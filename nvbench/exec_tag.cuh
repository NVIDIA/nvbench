/*
 *  Copyright 2021 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 with the LLVM exception
 *  (the "License"); you may not use this file except in compliance with
 *  the License.
 *
 *  You may obtain a copy of the License at
 *
 *      http://llvm.org/foundation/relicensing/LICENSE.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <nvbench/flags.cuh>

#include <cstdint>
#include <type_traits>

namespace nvbench::detail
{

// See the similarly named tags in nvbench::exec_tag:: for documentation.
enum class exec_flag : std::uint16_t
{
  none = 0x0,

  // Modifiers:
  timer         = 0x01, // KernelLauncher uses manual timing
  sync          = 0x02, // KernelLauncher has indicated that it will sync
  gpu           = 0x04, // Don't instantiate `measure_cpu_only`.
  no_gpu        = 0x08, // No GPU measurements should be instantiated.
  no_batch      = 0x10, // `measure_hot` will not be used.
  modifier_mask = 0xFF,

  // Measurement types to instantiate. Derived from modifiers.
  // Should not be exposed directly via nvbench::exec_tag::<...>.
  cold         = 0x0100, // measure_cold
  hot          = 0x0200, // measure_hot
  cpu_only     = 0x0400, // measure_cpu_only
  measure_mask = 0xFF00,
};

} // namespace nvbench::detail

NVBENCH_DECLARE_FLAGS(nvbench::detail::exec_flag)

namespace nvbench::exec_tag
{

namespace impl
{

struct tag_base
{};

template <typename ExecTag>
constexpr inline bool is_exec_tag_v = std::is_base_of_v<tag_base, ExecTag>;

/// Base class for exec_tag functionality.
/// This exists so that the `exec_flag`s can be embedded in a type with flag
/// semantics. This allows state::exec to only instantiate the measurements
/// that are actually used.
template <nvbench::detail::exec_flag Flags>
struct tag
    : std::integral_constant<nvbench::detail::exec_flag, Flags>
    , tag_base
{
  static constexpr nvbench::detail::exec_flag flags = Flags;

  template <nvbench::detail::exec_flag OFlags>
  constexpr auto operator|(tag<OFlags>) const
  {
    return tag<Flags | OFlags>{};
  }

  template <nvbench::detail::exec_flag OFlags>
  constexpr auto operator&(tag<OFlags>) const
  {
    return tag<Flags & OFlags>{};
  }

  constexpr auto operator~() const { return tag<~Flags>{}; }

  constexpr operator bool() const // NOLINT(google-explicit-constructor)
  {
    return Flags != nvbench::detail::exec_flag::none;
  }
};

using none_t          = tag<nvbench::detail::exec_flag::none>;
using timer_t         = tag<nvbench::detail::exec_flag::timer>;
using sync_t          = tag<nvbench::detail::exec_flag::sync>;
using gpu_t           = tag<nvbench::detail::exec_flag::gpu>;
using no_gpu_t        = tag<nvbench::detail::exec_flag::no_gpu>;
using no_batch_t      = tag<nvbench::detail::exec_flag::no_batch>;
using modifier_mask_t = tag<nvbench::detail::exec_flag::modifier_mask>;

using hot_t          = tag<nvbench::detail::exec_flag::hot>;
using cold_t         = tag<nvbench::detail::exec_flag::cold>;
using cpu_only_t     = tag<nvbench::detail::exec_flag::cpu_only>;
using measure_mask_t = tag<nvbench::detail::exec_flag::measure_mask>;

constexpr inline none_t none;
constexpr inline timer_t timer;
constexpr inline sync_t sync;
constexpr inline gpu_t gpu;
constexpr inline no_gpu_t no_gpu;
constexpr inline no_batch_t no_batch;
constexpr inline modifier_mask_t modifier_mask;

constexpr inline cold_t cold;
constexpr inline hot_t hot;
constexpr inline cpu_only_t cpu_only;
constexpr inline measure_mask_t measure_mask;

} // namespace impl

constexpr inline auto none = nvbench::exec_tag::impl::none;

/// Modifier used when only a portion of the KernelLauncher needs to be timed.
/// Useful for resetting state in-between timed kernel launches.
constexpr inline auto timer = nvbench::exec_tag::impl::timer | //
                              nvbench::exec_tag::impl::no_batch;

/// Modifier used to indicate that the KernelGenerator will perform CUDA
/// synchronizations. Without this flag such benchmarks will deadlock.
constexpr inline auto sync = nvbench::exec_tag::impl::sync | //
                             nvbench::exec_tag::impl::no_batch;

/// Modifier used to indicate that batched measurements should be disabled
constexpr inline auto no_batch = nvbench::exec_tag::impl::no_batch;

/// Optional optimization for CPU-only benchmarks. Requires that `set_is_cpu_only(true)`
/// is called when defining the benchmark. Passing this exec_tag will ensure that
/// GPU measurement code is not instantiated.
constexpr inline auto no_gpu = nvbench::exec_tag::impl::no_gpu;

/// Optional optimization for GPU benchmarks. Requires that `set_is_cpu_only(true)`
/// is NOT called when defining the benchmark. Passing this exec_tag will prevent unused CPU-only
/// measurement code from being instantiated.
constexpr inline auto gpu = nvbench::exec_tag::impl::gpu;

} // namespace nvbench::exec_tag
