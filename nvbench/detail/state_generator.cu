#include <nvbench/detail/state_generator.cuh>

#include <nvbench/benchmark_base.cuh>
#include <nvbench/named_values.cuh>
#include <nvbench/type_axis.cuh>

#include <nvbench/detail/transform_reduce.cuh>

#include <algorithm>
#include <cassert>
#include <functional>
#include <numeric>

namespace nvbench
{

namespace detail
{

// state_iterator ==============================================================

void state_iterator::add_axis(const nvbench::axis_base &axis)
{
  this->add_axis(axis.get_name(), axis.get_type(), axis.get_size());
}

void state_iterator::add_axis(std::string axis,
                              nvbench::axis_type type,
                              std::size_t size)
{
  m_indices.push_back({std::move(axis), type, std::size_t{0}, size});
}

[[nodiscard]] std::size_t state_iterator::get_number_of_states() const
{
  return nvbench::detail::transform_reduce(m_indices.cbegin(),
                                           m_indices.cend(),
                                           std::size_t{1},
                                           std::multiplies<>{},
                                           [](const axis_index &size_info) {
                                             return size_info.size;
                                           });
}

void state_iterator::init()
{
  m_current = 0;
  m_total   = this->get_number_of_states();
  for (axis_index &entry : m_indices)
  {
    entry.index = 0;
  }
}

[[nodiscard]] const std::vector<state_iterator::axis_index> &
state_iterator::get_current_indices() const
{
  return m_indices;
}

[[nodiscard]] bool state_iterator::iter_valid() const
{
  return m_current < m_total;
}

void state_iterator::next()
{
  for (axis_index &axis_info : m_indices)
  {
    axis_info.index += 1;
    if (axis_info.index >= axis_info.size)
    {
      axis_info.index = 0;
      continue; // carry the addition to the next entry in m_indices
    }
    break; // done
  }
  m_current += 1;
}

// state_generator =============================================================

state_generator::state_generator(const benchmark_base &bench)
    : m_benchmark(bench)
{}

void state_generator::build_axis_configs()
{
  const axes_metadata &axes = m_benchmark.get_axes();
  const std::vector<std::unique_ptr<axis_base>> &axes_vec = axes.get_axes();

  // Construct two state_generators:
  // - Only type_axis objects.
  // - Only non-type axes.
  state_iterator type_si;
  state_iterator non_type_si;

  // state_iterator initialization:
  {
    // stage the type axes in a vector to allow sorting:
    std::vector<std::reference_wrapper<const type_axis>> type_axes;
    type_axes.reserve(axes_vec.size());

    // Filter all axes by into type and non-type:
    std::for_each(axes_vec.cbegin(),
                  axes_vec.cend(),
                  [&non_type_si, &type_axes](const auto &axis) {
                    if (axis->get_type() == nvbench::axis_type::type)
                    {
                      type_axes.push_back(
                        std::cref(static_cast<const type_axis &>(*axis)));
                    }
                    else
                    {
                      non_type_si.add_axis(*axis);
                    }
                  });

    // Reverse sort type axes by index. This way the state_generator's cartesian
    // product of the type axes values will be enumerated in the same order as
    // nvbench::tl::cartesian_product<type_axes>. This is necessary to ensure
    // that the correct states are passed to the corresponding benchmark
    // instantiations.
    std::sort(type_axes.begin(),
              type_axes.end(),
              [](const auto &axis_1, const auto &axis_2) {
                return axis_1.get().get_axis_index() >
                       axis_2.get().get_axis_index();
              });

    std::for_each(type_axes.cbegin(),
                  type_axes.cend(),
                  [&type_si](const auto &axis) { type_si.add_axis(axis); });
  }

  // type_axis_configs generation:
  {
    m_type_axis_configs.clear();
    m_type_axis_configs.reserve(type_si.get_number_of_states());

    // Build type_axis_configs
    for (type_si.init(); type_si.iter_valid(); type_si.next())
    {
      auto &[config, active_mask] = m_type_axis_configs.emplace_back(
        std::make_pair(nvbench::named_values{}, true));

      // Reverse the indices so they're once again in the same order as
      // specified:
      auto indices = type_si.get_current_indices();
      std::reverse(indices.begin(), indices.end());

      for (const auto &axis_info : indices)
      {
        const auto &axis = axes.get_type_axis(axis_info.axis);
        if (!axis.get_is_active(axis_info.index))
        {
          active_mask = false;
        }

        config.set_string(axis_info.axis,
                          axis.get_input_string(axis_info.index));
      }
    } // type_si
  }   // type_axis_config generation

  // non_type_axis_config generation
  {
    m_non_type_axis_configs.clear();
    m_non_type_axis_configs.reserve(type_si.get_number_of_states());

    for (non_type_si.init(); non_type_si.iter_valid(); non_type_si.next())
    {
      auto &config = m_non_type_axis_configs.emplace_back();

      // Add non-type parameters to state:
      for (const auto &axis_info : non_type_si.get_current_indices())
      {
        switch (axis_info.type)
        {
          default:
          case axis_type::type:
            assert("unreachable." && false);
            break;

          case axis_type::int64:
            config.set_int64(
              axis_info.axis,
              axes.get_int64_axis(axis_info.axis).get_value(axis_info.index));
            break;

          case axis_type::float64:
            config.set_float64(
              axis_info.axis,
              axes.get_float64_axis(axis_info.axis).get_value(axis_info.index));
            break;

          case axis_type::string:
            config.set_string(
              axis_info.axis,
              axes.get_string_axis(axis_info.axis).get_value(axis_info.index));
            break;
        } // switch (type)
      }   // for (axis_info : current_indices)
    }     // for non_type_sg configs
  }       // non_type_axis_config generation
}

void state_generator::build_states()
{
  // Assemble states into a std::vector<std::vector<nvbench::state>>, where the
  // outer vector has one inner vector per type_config, and all configs in an
  // inner vector use the same type config. This should probably be wrapped up
  // into a nicer data structure, but organizing states in this way makes
  // matching up states to kernel_generator instantiations much easier during
  // dispatch.

  m_states.clear();
  m_states.reserve(m_type_axis_configs.size());
  for (const auto &[type_config, axis_mask] : m_type_axis_configs)
  {
    auto &inner_states = m_states.emplace_back();

    if (!axis_mask)
    { // Don't generate inner vector if the type config is masked out.
      continue;
    }

    inner_states.reserve(m_non_type_axis_configs.size());
    for (const auto &non_type_config : m_non_type_axis_configs)
    {
      nvbench::named_values config = type_config;
      config.append(non_type_config);
      inner_states.push_back(nvbench::state{m_benchmark, config});
    }
  }
}

std::vector<std::vector<nvbench::state>>
state_generator::create(const benchmark_base &bench)
{
  state_generator sg{bench};
  sg.build_axis_configs();
  sg.build_states();
  return std::move(sg.m_states);
}

} // namespace detail
} // namespace nvbench
