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

#include <nvbench/criterion_registry.cuh>
#include <nvbench/types.cuh>

#include "test_asserts.cuh"

void test_standard_criteria_exist()
{
  ASSERT(nvbench::criterion_registry::get("stdrel") != nullptr);
  ASSERT(nvbench::criterion_registry::get("entropy") != nullptr);
}

class custom_criterion : public nvbench::stopping_criterion
{
public:
  virtual void initialize(const nvbench::criterion_params &) override {}
  virtual void add_measurement(nvbench::float64_t /* measurement */) override {}
  virtual bool is_finished() override { return true; }
  virtual const params_description &get_params_description() const override
  {
    static const params_description desc{};
    return desc;
  }
};

void test_no_duplicates_are_allowed()
{
  bool exception_triggered = false;

  try {
    nvbench::stopping_criterion* custom = nvbench::criterion_registry::get("custom");
  } catch(...) {
    exception_triggered = true;
  }
  ASSERT(exception_triggered);

  std::unique_ptr<custom_criterion> custom_ptr = std::make_unique<custom_criterion>();
  custom_criterion* custom_raw = custom_ptr.get();
  ASSERT(nvbench::criterion_registry::register_criterion("custom", std::move(custom_ptr)));

  nvbench::stopping_criterion* custom = nvbench::criterion_registry::get("custom");
  ASSERT(custom_raw == custom);

  exception_triggered = false;
  try {
    nvbench::criterion_registry::register_criterion("custom", std::make_unique<custom_criterion>());
  } catch(...) {
    exception_triggered = true;
  }
  ASSERT(exception_triggered);
}

int main()
{
  test_standard_criteria_exist();
  test_no_duplicates_are_allowed();
}

