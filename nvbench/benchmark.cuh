#pragma once

#include <nvbench/benchmark_base.cuh>

#include <nvbench/axes_metadata.cuh>

namespace nvbench
{

// See benchmark_base for actual user API.
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
  // multiple benchmarks with the same name / definition together. That's not a
  // likely scenario, so we'll deal with that if it comes up.
  ~benchmark() override = default;

private:
  void do_set_type_axes_names(std::vector<std::string> names) override
  {
    m_axes.template set_type_axes_names<type_axes>(std::move(names));
  }
};

} // namespace nvbench
