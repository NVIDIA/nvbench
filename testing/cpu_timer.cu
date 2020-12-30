#include <nvbench/cpu_timer.cuh>

#include "test_asserts.cuh"

#include <chrono>
#include <thread>

void test_basic()
{
  using namespace std::literals::chrono_literals;

  nvbench::cpu_timer timer;

  timer.start();
  std::this_thread::sleep_for(250ms);
  timer.stop();

  ASSERT(timer.get_duration() > 0.25);
  ASSERT(timer.get_duration() < 0.50);
}

int main() { test_basic(); }
