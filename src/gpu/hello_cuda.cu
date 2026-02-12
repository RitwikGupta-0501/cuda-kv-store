#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

__global__ void hello_kernel() {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  printf("Hello from GPU thread %d\n", tid);
}

void print_gpu_info() {
  int device_count;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));

  printf("=== GPU Key-Value Store - System Info ===\n\n");
  printf("Found %d CUDA device(s)\n\n", device_count);

  for (int i = 0; i < device_count; i++) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, i));

    printf("Device %d: %s\n", i, prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  Total VRAM: %.2f GB\n", prop.totalGlobalMem / 1e9);
    printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("  Warp Size: %d\n", prop.warpSize);
    printf("\n");
  }

  // Get free memory
  size_t free_mem, total_mem;
  CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

  printf("Current VRAM Status:\n");
  printf("  Total: %.2f GB\n", total_mem / 1e9);
  printf("  Free: %.2f GB\n", free_mem / 1e9);
  printf("  Used: %.2f GB\n", (total_mem - free_mem) / 1e9);
  printf("\n");
}

int main() {
  print_gpu_info();

  printf("=== Testing CUDA Kernel ===\n");
  printf("Launching kernel with 2 blocks, 5 threads each...\n\n");

  // Launch a simple kernel
  hello_kernel<<<2, 5>>>();

  // Wait for GPU to finish
  CUDA_CHECK(cudaDeviceSynchronize());

  printf("\n✓ CUDA test successful!\n");

  return 0;
}
