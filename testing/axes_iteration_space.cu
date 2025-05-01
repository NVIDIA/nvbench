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

#include <nvbench/benchmark.cuh>
#include <nvbench/callable.cuh>
#include <nvbench/device_manager.cuh>
#include <nvbench/named_values.cuh>
#include <nvbench/state.cuh>
#include <nvbench/type_list.cuh>
#include <nvbench/type_strings.cuh>
#include <nvbench/types.cuh>

#include <fmt/format.h>

#include <algorithm>
#include <iterator>
#include <utility>
#include <variant>
#include <vector>

#include "test_asserts.cuh"

template <typename T>
std::vector<T> sort(std::vector<T> &&vec)
{
  std::sort(vec.begin(), vec.end());
  return std::move(vec);
}

void no_op_generator(nvbench::state &state)
{
  fmt::memory_buffer params;
  fmt::format_to(std::back_inserter(params), "Params:");
  const auto &axis_values = state.get_axis_values();
  for (const auto &name : sort(axis_values.get_names()))
  {
    std::visit(
      [&params, &name](const auto &value) {
        fmt::format_to(std::back_inserter(params), " {}: {}", name, value);
      },
      axis_values.get_value(name));
  }

  // Marking as skipped to signal that this state is run:
  state.skip(fmt::to_string(std::move(params)));
}
NVBENCH_DEFINE_CALLABLE(no_op_generator, no_op_callable);

template <typename KernelGenerator, typename TypeAxes = nvbench::type_list<>>
struct rezippable_benchmark final : public nvbench::benchmark_base
{
  using kernel_generator = KernelGenerator;
  using type_axes        = TypeAxes;
  using type_configs     = nvbench::tl::cartesian_product<type_axes>;

  static constexpr std::size_t num_type_configs =
    nvbench::tl::size<type_configs>{};

  rezippable_benchmark()
      : benchmark_base(type_axes{})
  {}

private:
  std::unique_ptr<benchmark_base> do_clone() const final
  {
    return std::make_unique<rezippable_benchmark>();
  }

  void do_set_type_axes_names(std::vector<std::string> names) final
  {
    m_axes.set_type_axes_names(std::move(names));
  }

  void do_run() final
  {
    nvbench::runner<rezippable_benchmark> runner{*this};
    runner.generate_states();
    runner.run();
  }
};

template <typename Integer, typename Float, typename Other>
void template_no_op_generator(nvbench::state &state,
                              nvbench::type_list<Integer, Float, Other>)
{
  ASSERT(nvbench::type_strings<Integer>::input_string() ==
         state.get_string("Integer"));
  ASSERT(nvbench::type_strings<Float>::input_string() ==
         state.get_string("Float"));
  ASSERT(nvbench::type_strings<Other>::input_string() ==
         state.get_string("Other"));

  // Enum params using non-templated version:
  no_op_generator(state);
}
NVBENCH_DEFINE_CALLABLE_TEMPLATE(template_no_op_generator,
                                 template_no_op_callable);

void test_zip_axes()
{
  using benchmark_type = nvbench::benchmark<no_op_callable>;
  benchmark_type bench;
  bench.set_devices(nvbench::device_manager::get().get_devices());
  bench.add_zip_axes(nvbench::float64_axis("F64 Axis", {0., .1, .25, .5, 1.}),
                     nvbench::int64_axis("I64 Axis", {1, 3, 2, 4, 5}));

  const auto num_devices = std::max(std::size_t(1), bench.get_devices().size());
  ASSERT_MSG(bench.get_config_count() == 5 * num_devices,
             "Got {}, expected {}",
             bench.get_config_count(),
             5 * bench.get_devices().size());

  bench.set_devices(std::vector<int>{});
  ASSERT_MSG(bench.get_config_count() == 5, "Got {}, expected {}", bench.get_config_count(), 5);
}

void test_zip_unequal_length()
{
  using benchmark_type = nvbench::benchmark<no_op_callable>;
  benchmark_type bench;

  ASSERT_THROWS_ANY(
    bench.add_zip_axes(nvbench::float64_axis("F64 Axis", {0., .1, .25, .5, 1.}),
                       nvbench::int64_axis("I64 Axis", {1, 3, 2})));
}

void test_zip_clone()
{
  using benchmark_type = nvbench::benchmark<no_op_callable>;
  benchmark_type bench;
  bench.set_devices(std::vector<int>{});
  bench.add_int64_power_of_two_axis("I64 POT Axis", {10, 20});
  bench.add_int64_axis("I64 Axis", {10, 20});
  bench.add_zip_axes(nvbench::string_axis("Strings",
                                          {"string a", "string b", "string c"}),
                     nvbench::float64_axis("F64 Axis", {0., .1, .25}));

  const auto expected_count = bench.get_config_count();

  std::unique_ptr<nvbench::benchmark_base> clone_base = bench.clone();
  ASSERT(clone_base.get() != nullptr);

  ASSERT_MSG(expected_count == clone_base->get_config_count(),
             "Got {}",
             clone_base->get_config_count());

  auto *clone = dynamic_cast<benchmark_type *>(clone_base.get());
  ASSERT(clone != nullptr);

  ASSERT(bench.get_name() == clone->get_name());

  const auto &ref_axes   = bench.get_axes().get_axes();
  const auto &clone_axes = clone->get_axes().get_axes();
  ASSERT(ref_axes.size() == clone_axes.size());
  for (std::size_t i = 0; i < ref_axes.size(); ++i)
  {
    const nvbench::axis_base *ref_axis   = ref_axes[i].get();
    const nvbench::axis_base *clone_axis = clone_axes[i].get();
    ASSERT(ref_axis != nullptr);
    ASSERT(clone_axis != nullptr);
    ASSERT(ref_axis->get_name() == clone_axis->get_name());
    ASSERT(ref_axis->get_type() == clone_axis->get_type());
    ASSERT(ref_axis->get_size() == clone_axis->get_size());
    for (std::size_t j = 0; j < ref_axis->get_size(); ++j)
    {
      ASSERT(ref_axis->get_input_string(j) == clone_axis->get_input_string(j));
      ASSERT(ref_axis->get_description(j) == clone_axis->get_description(j));
    }
  }

  ASSERT(clone->get_states().empty());
}

struct under_diag final : nvbench::user_axis_space
{
  under_diag(std::vector<std::size_t> input_indices)
      : nvbench::user_axis_space(std::move(input_indices))
  {}

  mutable std::size_t x_pos   = 0;
  mutable std::size_t y_pos   = 0;
  mutable std::size_t x_start = 0;

  nvbench::detail::axis_space_iterator do_get_iterator(axes_info info) const
  {
    // generate our increment function
    auto adv_func = [&, info](std::size_t &inc_index,
                              std::size_t /*len*/) -> bool {
      inc_index++;
      x_pos++;
      if (x_pos == info[0].size)
      {
        x_pos = ++x_start;
        y_pos = x_start;
        return true;
      }
      return false;
    };

    // our update function
    auto diag_under =
      [&, info](std::size_t,
                std::vector<nvbench::detail::axis_index>::iterator start,
                std::vector<nvbench::detail::axis_index>::iterator end) {
        start->index = x_pos;
        end->index   = y_pos;
      };

    const size_t iteration_length = ((info[0].size * (info[1].size + 1)) / 2);
    return nvbench::detail::axis_space_iterator(info,
                                                iteration_length,
                                                adv_func,
                                                diag_under);
  }

  std::size_t do_get_size(const axes_info &info) const
  {
    return ((info[0].size * (info[1].size + 1)) / 2);
  }

  std::size_t do_get_active_count(const axes_info &info) const
  {
    return ((info[0].size * (info[1].size + 1)) / 2);
  }

  std::unique_ptr<nvbench::iteration_space_base> do_clone() const
  {
    return std::make_unique<under_diag>(*this);
  }
};

void test_user_axes()
{
  using benchmark_type = rezippable_benchmark<no_op_callable>;
  benchmark_type bench;
  bench.set_devices(nvbench::device_manager::get().get_devices());
  bench.add_user_iteration_axes(
    [](auto... args) -> std::unique_ptr<nvbench::iteration_space_base> {
      return std::make_unique<under_diag>(args...);
    },
    nvbench::float64_axis("F64 Axis", {0., .1, .25, .5, 1.}),
    nvbench::int64_axis("I64 Axis", {1, 3, 2, 4, 5}));

  const auto num_devices = std::max(std::size_t(1), bench.get_devices().size());
  ASSERT_MSG(bench.get_config_count() == 15 * num_devices, "Got {}", bench.get_config_count());

  bench.set_devices(std::vector<int>{});
  ASSERT_MSG(bench.get_config_count() == 15, "Got {}", bench.get_config_count());
}

int main()
{
  test_zip_axes();
  test_zip_unequal_length();
  test_zip_clone();

  test_user_axes();
}
