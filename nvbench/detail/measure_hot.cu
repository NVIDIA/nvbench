#include <nvbench/detail/measure_hot.cuh>

#include <nvbench/state.cuh>
#include <nvbench/summary.cuh>

namespace nvbench
{

namespace detail
{

void measure_hot_base::generate_summaries()
{
  {
    auto &summary = m_state.add_summary("Number of Trials (Hot)");
    summary.set_string("short_name", "Hot Trials");
    summary.set_string("description",
                       "Number of kernel executions in hot time measurements.");
    summary.set_int64("value", m_num_trials);
  }

  {
    auto &summary = m_state.add_summary("Average GPU Time (Hot)");
    summary.set_string("hint", "duration");
    summary.set_string("short_name", "Hot GPU");
    summary.set_string("description",
                       "Average back-to-back kernel execution time as measured "
                       "by CUDA events.");
    summary.set_float64("value", m_cuda_time / m_num_trials);
  }

  {
    auto &summary = m_state.add_summary("Average CPU Time (Hot)");
    summary.set_string("hint", "duration");
    summary.set_string("short_name", "Hot CPU");
    summary.set_string("description",
                       "Average back-to-back kernel execution time observed "
                       "from host.");
    summary.set_float64("value", m_cpu_time / m_num_trials);
  }
}

} // namespace detail

} // namespace nvbench
