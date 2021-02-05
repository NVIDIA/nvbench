#include <nvbench/named_values.cuh>

#include "test_asserts.cuh"

#include <algorithm>

void test_empty()
{
  nvbench::named_values vals;
  ASSERT(vals.get_size() == 0);
  ASSERT(vals.get_names().size() == 0);
  ASSERT(vals.has_value("Nope") == false);
  ASSERT_THROWS_ANY([[maybe_unused]] auto val = vals.get_value("Nope"));
  ASSERT_THROWS_ANY([[maybe_unused]] auto type = vals.get_type("Nope"));
  // Removing non-existent entries shouldn't cause a problem:
  vals.remove_value("Nope");
}

void test_basic()
{
  auto sort = [](auto &&vec) {
    std::sort(vec.begin(), vec.end());
    return std::forward<decltype(vec)>(vec);
  };

  nvbench::named_values vals;
  vals.set_int64("Int", 32);
  vals.set_float64("Float", 34.5);
  vals.set_string("String", "string!");
  vals.set_value("IntVar", {nvbench::int64_t{36}});

  std::vector<std::string> names{"Float", "Int", "IntVar", "String"};

  ASSERT(vals.get_size() == 4);
  ASSERT(sort(vals.get_names()) == names);

  ASSERT(vals.has_value("Float"));
  ASSERT(vals.has_value("Int"));
  ASSERT(vals.has_value("IntVar"));
  ASSERT(vals.has_value("String"));

  ASSERT(std::get<nvbench::float64_t>(vals.get_value("Float")) == 34.5);
  ASSERT(std::get<nvbench::int64_t>(vals.get_value("Int")) == 32);
  ASSERT(std::get<nvbench::int64_t>(vals.get_value("IntVar")) == 36);
  ASSERT(std::get<std::string>(vals.get_value("String")) == "string!");

  ASSERT(vals.get_type("Float") == nvbench::named_values::type::float64);
  ASSERT(vals.get_type("Int") == nvbench::named_values::type::int64);
  ASSERT(vals.get_type("IntVar") == nvbench::named_values::type::int64);
  ASSERT(vals.get_type("String") == nvbench::named_values::type::string);

  ASSERT(vals.get_int64("Int") == 32);
  ASSERT(vals.get_int64("IntVar") == 36);
  ASSERT_THROWS_ANY([[maybe_unused]] auto v = vals.get_int64("Float"));
  ASSERT_THROWS_ANY([[maybe_unused]] auto v = vals.get_int64("String"));

  ASSERT(vals.get_float64("Float") == 34.5);
  ASSERT_THROWS_ANY([[maybe_unused]] auto v = vals.get_float64("Int"));
  ASSERT_THROWS_ANY([[maybe_unused]] auto v = vals.get_float64("IntVar"));
  ASSERT_THROWS_ANY([[maybe_unused]] auto v = vals.get_float64("String"));

  ASSERT(vals.get_string("String") == "string!");
  ASSERT_THROWS_ANY([[maybe_unused]] auto v = vals.get_string("Int"));
  ASSERT_THROWS_ANY([[maybe_unused]] auto v = vals.get_string("IntVar"));
  ASSERT_THROWS_ANY([[maybe_unused]] auto v = vals.get_string("Float"));

  vals.remove_value("IntVar");
  names = {"Float", "Int", "String"};

  ASSERT(vals.get_size() == 3);
  ASSERT(sort(vals.get_names()) == names);

  ASSERT(!vals.has_value("IntVar"));
  ASSERT(vals.has_value("Float"));
  ASSERT(vals.has_value("Int"));
  ASSERT(vals.has_value("String"));

  vals.clear();
  names = {};

  ASSERT(vals.get_size() == 0);
  ASSERT(sort(vals.get_names()) == names);

  ASSERT(!vals.has_value("IntVar"));
  ASSERT(!vals.has_value("Float"));
  ASSERT(!vals.has_value("Int"));
  ASSERT(!vals.has_value("String"));
}

int main()
{
  test_empty();
  test_basic();
}
