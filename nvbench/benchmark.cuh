#pragma once

#include <nvbench/benchmark_base.cuh>

#include <nvbench/axes_metadata.cuh>
#include <nvbench/type_list.cuh>

#include <string>
#include <vector>

namespace nvbench
{

/**
 * Holds a complete benchmark specification: a KernelGenerator and parameter
 * axes.
 *
 * Creation and configuration of this class is documented in the NVBench
 * [README](../README.md) file. Refer to that for usage details.
 *
 * This class is purposefully kept small to reduce the amount of template code
 * generated for each benchmark. Most of the functionality is implemented in
 * nvbench::benchmark_base -- this class only holds type aliases related to the
 * `KernelGenerator` and `TypeAxes` parameters, and exposes them to
 * `benchmark_base` through a private virtual API.
 *
 * Delegates responsibilities to the following classes:
 * - nvbench::benchmark_base: all non-templated benchmark handling.
 *
 * @tparam KernelGenerator See the [README](../README.md).
 * @tparam TypeAxes A `nvbench::type_list` of `nvbench::type_list`s. See the
 * [README](../README.md) for more details.
 */
template <typename KernelGenerator, typename TypeAxes>
struct benchmark final : public benchmark_base
{
  using kernel_generator = KernelGenerator;
  using type_axes        = TypeAxes;
  using type_configs     = nvbench::tl::cartesian_product<type_axes>;

  static constexpr std::size_t num_type_configs =
    nvbench::tl::size<type_configs>{};

  using benchmark_base::benchmark_base;

  // Note that this inline virtual dtor may cause vtable issues if linking
  // benchmark TUs together. That's not a likely scenario, so we'll deal with
  // that if it comes up.
  ~benchmark() override = default;

private:
  void do_set_type_axes_names(std::vector<std::string> names) override
  {
    m_axes.template set_type_axes_names<type_axes>(std::move(names));
  }
};

} // namespace nvbench
