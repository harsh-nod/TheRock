// vector_add kernel used as Mirage JIT translation test fixture.
// Compiled with: hipcc --offload-arch=gfx1201 --genco -o vector_add.co source_excerpt.hip
#include <hip/hip_runtime.h>

extern "C" __global__ void vector_add(float* out, const float* a,
                                       const float* b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] + b[i];
  }
}
