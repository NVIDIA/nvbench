#include <nvbench/benchmark_base.cuh>

#include <nvbench/device_manager.cuh>

namespace nvbench
{

benchmark_base::benchmark_base()
    : m_devices(nvbench::device_manager::get().get_devices())
{}

benchmark_base::~benchmark_base() = default;

std::unique_ptr<benchmark_base> benchmark_base::clone() const
{
  auto result = this->do_clone();

  // Do not copy states.
  result->m_name    = m_name;
  result->m_axes    = m_axes;
  result->m_devices = m_devices;

  result->m_min_samples = m_min_samples;
  result->m_min_time    = m_min_time;
  result->m_max_noise   = m_max_noise;

  result->m_skip_time = m_skip_time;
  result->m_timeout   = m_timeout;

  return std::move(result);
}

void benchmark_base::set_devices(std::vector<int> device_ids)
{
  std::vector<device_info> devices;
  devices.reserve(device_ids.size());
  for (int dev_id : device_ids)
  {
    devices.emplace_back(dev_id);
  }
  this->set_devices(std::move(devices));
}

void benchmark_base::add_device(int device_id)
{
  this->add_device(device_info{device_id});
}

} // namespace nvbench
