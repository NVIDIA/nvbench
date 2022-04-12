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

#include <nvbench/nvbench.cuh>

// Grab some testing kernels from NVBench:
#include <nvbench/test_kernels.cuh>

// Thrust vectors simplify memory management:
#include <thrust/device_vector.h>

#include <random>

//==============================================================================
// Multiple parameters:
// Varies block_size and num_blocks while invoking a naive copy of 256 MiB worth
// of int32_t.
void copy_sweep_grid_shape(nvbench::state &state)
{
  // Get current parameters:
  const int block_size = static_cast<int>(state.get_int64("BlockSize"));
  const int num_blocks = static_cast<int>(state.get_int64("NumBlocks"));

  // Number of int32s in 256 MiB:
  const std::size_t num_values = 256 * 1024 * 1024 / sizeof(nvbench::int32_t);

  // Report throughput stats:
  state.add_element_count(num_values);
  state.add_global_memory_reads<nvbench::int32_t>(num_values);
  state.add_global_memory_writes<nvbench::int32_t>(num_values);

  // Allocate device memory:
  thrust::device_vector<nvbench::int32_t> in(num_values, 0);
  thrust::device_vector<nvbench::int32_t> out(num_values, 0);

  state.exec(
    [block_size,
     num_blocks,
     num_values,
     in_ptr  = thrust::raw_pointer_cast(in.data()),
     out_ptr = thrust::raw_pointer_cast(out.data())](nvbench::launch &launch) {
      nvbench::copy_kernel<<<num_blocks, block_size, 0, launch.get_stream()>>>(
        in_ptr,
        out_ptr,
        num_values);
    });
}
void naive_copy_sweep_grid_shape(nvbench::state &state)
{
  copy_sweep_grid_shape(state);
}
void tied_copy_sweep_grid_shape(nvbench::state &state)
{
  copy_sweep_grid_shape(state);
}

//==============================================================================
// Naive iteration of both the BlockSize and NumBlocks axes.
// Will generate the full cartesian product of the two axes for a total of
// 16 invocations of copy_sweep_grid_shape.
NVBENCH_BENCH(naive_copy_sweep_grid_shape)
  // Full combinatorial of Every power of two from  64->1024:
  .add_int64_axis("BlockSize", {32, 64, 128, 256})
  .add_int64_axis("NumBlocks", {1024, 512, 256, 128});

//==============================================================================
// Zipped iteration of BlockSize and NumBlocks axes.
// Will generate only 4 invocations of copy_sweep_grid_shape
NVBENCH_BENCH(tied_copy_sweep_grid_shape)
  // Every power of two from  64->1024:
  .add_zip_axes(nvbench::int64_axis{"BlockSize", {32, 64, 128, 256}},
                nvbench::int64_axis{"NumBlocks", {1024, 512, 256, 128}});

//==============================================================================
// under_diag:
// Custom iterator that only searches the `X` locations of two axi
// [- - - - X]
// [- - - X X]
// [- - X X X]
// [- X X X X]
// [X X X X X]
//
struct under_diag final : nvbench::user_axis_space
{
  under_diag(std::vector<std::size_t> input_indices,
             std::vector<std::size_t> output_indices)
      : nvbench::user_axis_space(std::move(input_indices),
                                 std::move(output_indices))
  {}

  mutable std::size_t x_pos   = 0;
  mutable std::size_t y_pos   = 0;
  mutable std::size_t x_start = 0;

  nvbench::detail::axis_space_iterator do_iter(axes_info info) const
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
    std::vector<std::size_t> locs = m_output_indices;
    auto diag_under =
      [&, locs, info](std::size_t,
                      std::vector<nvbench::detail::axis_index> &indices) {
        nvbench::detail::axis_index temp = info[0];
        temp.index                       = x_pos;
        indices[locs[0]]                 = temp;

        temp             = info[1];
        temp.index       = y_pos;
        indices[locs[1]] = temp;
      };

    const size_t iteration_length = ((info[0].size * (info[1].size + 1)) / 2);
    return nvbench::detail::make_space_iterator(2,
                                                iteration_length,
                                                adv_func,
                                                diag_under);
  }

  std::size_t do_size(const axes_info &info) const
  {
    return ((info[0].size * (info[1].size + 1)) / 2);
  }

  std::size_t do_valid_count(const axes_info &info) const
  {
    return ((info[0].size * (info[1].size + 1)) / 2);
  }

  std::unique_ptr<nvbench::axis_space_base> do_clone() const
  {
    return std::make_unique<under_diag>(*this);
  }
};

void user_copy_sweep_grid_shape(nvbench::state &state)
{
  copy_sweep_grid_shape(state);
}
NVBENCH_BENCH(user_copy_sweep_grid_shape)
  .add_user_iteration_axes(
    [](auto... args) -> std::unique_ptr<nvbench::axis_space_base> {
      return std::make_unique<under_diag>(args...);
    },
    nvbench::int64_axis("BlockSize", {64, 128, 256, 512, 1024}),
    nvbench::int64_axis("NumBlocks", {1024, 521, 256, 128, 64}));

//==============================================================================
// gauss:
// Custom iteration space that uses a gauss distribution to
// sample the points near the middle of the index space
//
struct gauss final : nvbench::user_axis_space
{

  gauss(std::vector<std::size_t> input_indices,
        std::vector<std::size_t> output_indices)
      : nvbench::user_axis_space(std::move(input_indices),
                                 std::move(output_indices))
  {}

  nvbench::detail::axis_space_iterator do_iter(axes_info info) const
  {
    const double mid_point = static_cast<double>((info[0].size / 2));

    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> d{mid_point, 2};

    const size_t iteration_length = info[0].size;
    std::vector<std::size_t> gauss_indices(iteration_length);
    for (auto &g : gauss_indices)
    {
      auto v = std::min(static_cast<double>(info[0].size), d(gen));
      v      = std::max(0.0, v);
      g      = static_cast<std::size_t>(v);
    }

    // our update function
    std::vector<std::size_t> locs = m_output_indices;
    auto gauss_func               = [=](std::size_t index,
                          std::vector<nvbench::detail::axis_index> &indices) {
      nvbench::detail::axis_index temp = info[0];
      temp.index                       = gauss_indices[index];
      indices[locs[0]]                 = temp;
    };

    return nvbench::detail::make_space_iterator(1,
                                                iteration_length,
                                                gauss_func);
  }

  std::size_t do_size(const axes_info &info) const { return info[0].size; }

  std::size_t do_valid_count(const axes_info &info) const
  {
    return info[0].size;
  }

  std::unique_ptr<axis_space_base> do_clone() const
  {
    return std::make_unique<gauss>(*this);
  }
};
//==============================================================================
// Dual parameter sweep:
void dual_float64_axis(nvbench::state &state)
{
  const auto duration_A = state.get_float64("Duration_A");
  const auto duration_B = state.get_float64("Duration_B");

  state.exec([duration_A, duration_B](nvbench::launch &launch) {
    nvbench::sleep_kernel<<<1, 1, 0, launch.get_stream()>>>(duration_A +
                                                            duration_B);
  });
}
NVBENCH_BENCH(dual_float64_axis)
  .add_user_iteration_axes(
    [](auto... args) -> std::unique_ptr<nvbench::axis_space_base> {
      return std::make_unique<gauss>(args...);
    },
    nvbench::float64_axis("Duration_A", nvbench::range(0., 1e-4, 1e-5)))
  .add_user_iteration_axes(
    [](auto... args) -> std::unique_ptr<nvbench::axis_space_base> {
      return std::make_unique<gauss>(args...);
    },
    nvbench::float64_axis("Duration_B", nvbench::range(0., 1e-4, 1e-5)));
