#include <nvbench/detail/state_generator.cuh>

#include <nvbench/benchmark_base.cuh>
#include <nvbench/named_values.cuh>
#include <nvbench/type_axis.cuh>

#include <algorithm>
#include <functional>
#include <numeric>

#include <cassert>
#include <functional>
#include <numeric>

namespace nvbench
{

namespace detail
{

std::vector<std::vector<nvbench::state>>
state_generator::create(const benchmark_base &bench)
{
  // Assemble states into a std::vector<std::vector<nvbench::state>>, where the
  // outer vector has one inner vector per type_config, and all configs in an
  // inner vector use the same type config. This should probably be wrapped up
  // into a nicer data structure, but organizing states in this way makes
  // matching up states to kernel_generator instantiations much easier during
  // dispatch.

  const axes_metadata &axes = bench.get_axes();
  // vector of all axes:
  const std::vector<std::unique_ptr<const axis_base>> &axes_vec =
    axes.get_axes();

  // Construct two state_generators:
  // - Only type_axis objects,
  // - Only non-type axes.
  state_generator type_sg;
  state_generator non_type_sg;
  {
    // stage the type axes in a vector to allow sorting:
    std::vector<std::reference_wrapper<const type_axis>> type_axes;
    type_axes.reserve(axes_vec.size());

    // Filter all axes by into type and non-type:
    std::for_each(axes_vec.cbegin(),
                  axes_vec.cend(),
                  [&non_type_sg, &type_axes](const auto &axis) {
                    if (axis->get_type() == nvbench::axis_type::type)
                    {
                      type_axes.push_back(
                        std::cref(static_cast<const type_axis &>(*axis)));
                    }
                    else
                    {
                      non_type_sg.add_axis(*axis);
                    }
                  });

    // Reverse sort type axes by index. This way the state_generator's cartesian
    // product of the type axes values will be enumerated in the same order as
    // nvbench::tl::cartesian_product<type_axes>.
    std::sort(type_axes.begin(),
              type_axes.end(),
              [](const auto &axis_1, const auto &axis_2) {
                return axis_1.get().get_axis_index() >
                       axis_2.get().get_axis_index();
              });

    std::for_each(type_axes.cbegin(),
                  type_axes.cend(),
                  [&type_sg](const auto &axis) { type_sg.add_axis(axis); });
  }

  std::vector<std::vector<nvbench::state>> result;
  {
    const std::size_t num_type_configs     = type_sg.get_number_of_states();
    const std::size_t num_non_type_configs = non_type_sg.get_number_of_states();

    result.reserve(num_type_configs);

    // name / input_string pairs for type axes.
    // Stored in named_values to simplify initializing the state objects.
    nvbench::named_values type_config;

    // Iterate through type axis combinations:
    for (type_sg.init(); type_sg.iter_valid(); type_sg.next())
    {
      // Construct map of current type axis parameters:
      type_config.clear();

      // Reverse the type axes so they're once again in the same order as
      // specified:
      auto indices = type_sg.get_current_indices();
      std::reverse(indices.begin(), indices.end());
      for (const auto &axis_info : indices)
      {
        type_config.set_string(
          axis_info.axis,
          axes.get_type_axis(axis_info.axis).get_input_string(axis_info.index));
      }

      // Create the inner vector of states for the current type_config:
      auto &states = result.emplace_back();
      states.reserve(num_non_type_configs);
      for (non_type_sg.init(); non_type_sg.iter_valid(); non_type_sg.next())
      {
        // Initialize each state with the current type_config:
        nvbench::state state{bench, type_config};
        // Add non-type parameters to state:
        for (const axis_index &axis_info : non_type_sg.get_current_indices())
        {
          switch (axis_info.type)
          {
            default:
            case axis_type::type:
              assert("unreachable." && false);
              break;

            case axis_type::int64:
              state.m_axis_values.set_int64(
                axis_info.axis,
                axes.get_int64_axis(axis_info.axis).get_value(axis_info.index));
              break;

            case axis_type::float64:
              state.m_axis_values.set_float64(
                axis_info.axis,
                axes.get_float64_axis(axis_info.axis)
                  .get_value(axis_info.index));
              break;

            case axis_type::string:
              state.m_axis_values.set_string(
                axis_info.axis,
                axes.get_string_axis(axis_info.axis).get_value(axis_info.index));
              break;
          } // switch (type)
        }   // for (axis_info : current_indices)
        states.push_back(std::move(state));
      } // for non_type_sg configs
    }   // for type_sg configs
  }     // scope break

  // phew.
  return result;
}

std::size_t state_generator::get_number_of_states() const
{
  return std::transform_reduce(m_indices.cbegin(),
                               m_indices.cend(),
                               std::size_t{1},
                               std::multiplies<>{},
                               [](const axis_index &size_info) {
                                 return size_info.size;
                               });
}

void state_generator::init()
{
  m_current = 0;
  m_total   = this->get_number_of_states();
  for (axis_index &entry : m_indices)
  {
    entry.index = 0;
  }
}

bool state_generator::iter_valid() const { return m_current < m_total; }

void state_generator::next()
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

} // namespace detail
} // namespace nvbench
