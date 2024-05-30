#include <nvbench/cuda_call.cuh>

#include "test_asserts.cuh"


namespace
{
    __global__ void multiply5(const int32_t* __restrict__ a, int32_t* __restrict__ b)
    {
      const auto id = blockIdx.x * blockDim.x + threadIdx.x;
      b[id] = 5 * a[id];
    }
}

int main()
{ 
  multiply5<<<256, 256>>>(nullptr, nullptr);

  try
  {
    NVBENCH_CUDA_CALL(cudaStreamSynchronize(0));
    ASSERT(false);
  }
  catch (const std::runtime_error &)
  {
    ASSERT(cudaGetLastError() == cudaError_t::cudaSuccess);
  }

  return 0;
}
