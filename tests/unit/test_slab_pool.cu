#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../../include/warpkv/slab_pool.cuh"
#include <vector>

using namespace warpkv;

__global__ void bump_alloc_kernel(SlabPool<half> pool, uint32_t* out_indices, int num_threads) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_threads) {
        // Lock-free bump allocation
        uint32_t slab_idx = atomicAdd(pool.atomic_counter, 1);
        out_indices[tid] = slab_idx;

        // Write a predictable value to the vector to test coalescing/integrity
        if (slab_idx < pool.max_capacity) {
            half* vec = pool.get_vector(slab_idx);
            for (uint32_t i = 0; i < pool.vector_dim; ++i) {
                vec[i] = __float2half((float)tid);
            }
        }
    }
}

TEST(SlabPoolTest, BumpAllocationAndIntegrity) {
    const uint32_t max_capacity = 10000;
    const uint32_t vector_dim = 128;
    const int num_threads = 5000;

    SlabPool<half> pool;
    pool.vector_dim = vector_dim;
    pool.max_capacity = max_capacity;

    cudaMalloc(&pool.data, max_capacity * vector_dim * sizeof(half));
    cudaMalloc(&pool.atomic_counter, sizeof(uint32_t));
    cudaMemset(pool.atomic_counter, 0, sizeof(uint32_t));

    uint32_t* d_out_indices;
    cudaMalloc(&d_out_indices, num_threads * sizeof(uint32_t));

    int block_size = 256;
    int grid_size = (num_threads + block_size - 1) / block_size;

    bump_alloc_kernel<<<grid_size, block_size>>>(pool, d_out_indices, num_threads);
    cudaDeviceSynchronize();

    uint32_t final_count;
    cudaMemcpy(&final_count, pool.atomic_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(final_count, num_threads);

    // Validate indices are unique and in range 0..4999
    std::vector<uint32_t> h_indices(num_threads);
    cudaMemcpy(h_indices.data(), d_out_indices, num_threads * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    std::vector<bool> seen(num_threads, false);
    for (int i = 0; i < num_threads; ++i) {
        ASSERT_LT(h_indices[i], num_threads);
        EXPECT_FALSE(seen[h_indices[i]]);
        seen[h_indices[i]] = true;
    }

    cudaFree(pool.data);
    cudaFree(pool.atomic_counter);
    cudaFree(d_out_indices);
}
