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

#pragma once

#include <nvbench/types.cuh>
#include <nvbench/named_values.cuh>

#include <unordered_map>
#include <string>

namespace nvbench
{

namespace detail 
{

constexpr nvbench::float64_t compat_min_time() { return 0.5; }    // 0.5 seconds
constexpr nvbench::float64_t compat_max_noise() { return 0.005; } // 0.5% relative standard deviation

} // namespace detail

/**
 * Stores all the parameters for stopping criterion in use
 */
class criterion_params
{
  nvbench::named_values m_named_values;
public:

  void set_int64(std::string name, nvbench::int64_t value);
  void set_float64(std::string name, nvbench::float64_t value);
  void set_string(std::string name, std::string value);

  [[nodiscard]] bool has_value(const std::string &name) const;
  [[nodiscard]] nvbench::int64_t get_int64(const std::string &name) const;
  [[nodiscard]] nvbench::float64_t get_float64(const std::string &name) const;
};

/**
 * Stopping criterion interface
 */
class stopping_criterion
{
protected:
  std::string m_name;
  criterion_params m_params;

public:
  explicit stopping_criterion(std::string name) : m_name(std::move(name)) { }

  [[nodiscard]] const std::string &get_name() const { return m_name; }

  /**
   * Initialize the criterion with the given parameters
   *
   * This method is called once per benchmark run, before any measurements are provided.
   */
  virtual void initialize(const criterion_params &params) = 0;

  /**
   * Add the latest measurement to the criterion
   */
  virtual void add_measurement(nvbench::float64_t measurement) = 0;

  /**
   * Check if the criterion has been met for all measurements processed by `add_measurement`
   */
  virtual bool is_finished() = 0;

  using params_description = std::vector<std::pair<std::string, nvbench::named_values::type>>;

  /**
   * Return the parameter names and types for this criterion
   */
  virtual const params_description &get_params_description() const = 0;
};

} // namespace nvbench
