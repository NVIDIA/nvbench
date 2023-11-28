/*
 *  Copyright 2023 NVIDIA Corporation
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

#include <nvbench/stopping_criterion.cuh>


namespace nvbench
{

void criterion_params::set_int64(std::string name, nvbench::int64_t value)
{
  if (m_named_values.has_value(name)) 
  {
    m_named_values.remove_value(name);
  }

  m_named_values.set_int64(name, value);
}

void criterion_params::set_float64(std::string name, nvbench::float64_t value)
{
  if (m_named_values.has_value(name)) 
  {
    m_named_values.remove_value(name);
  }

  m_named_values.set_float64(name, value);
}

void criterion_params::set_string(std::string name, std::string value)
{
  if (m_named_values.has_value(name)) 
  {
    m_named_values.remove_value(name);
  }

  m_named_values.set_string(name, std::move(value));
}

bool criterion_params::has_value(const std::string &name) const
{
  if (name == "max-noise" || name == "min-time")
  { // compat
    return true;
  }
  return m_named_values.has_value(name);
}

nvbench::int64_t criterion_params::get_int64(const std::string &name) const
{
  return m_named_values.get_int64(name);
}

nvbench::float64_t criterion_params::get_float64(const std::string &name) const
{
  if (!m_named_values.has_value(name)) 
  {
    if (name == "max-noise")
    { // compat
      return nvbench::detail::compat_max_noise();
    }
    else if (name == "min-time")
    { // compat
      return nvbench::detail::compat_min_time();
    }
  }
  return m_named_values.get_float64(name);
}

} // namespace nvbench::detail
