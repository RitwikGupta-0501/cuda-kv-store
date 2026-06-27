#include "cuckoo_insert.h"
#include <stdexcept>
#include <cstring>

namespace warpkv {

// Host-side wrapper: allocate temporary buffers and launch kernel
void warp_insert_batch(
    BucketTable table,
    StashQueue* d_stash,
    const InsertBatch& batch,
    cudaStream_t stream) {

    if (batch.num_keys == 0) return;

    uint32_t* d_keys = nullptr;
    uint32_t* d_values = nullptr;
    InsertStatus* d_statuses = nullptr;
    uint32_t* d_hops = nullptr;

    size_t keys_size = batch.num_keys * sizeof(uint32_t);
    size_t statuses_size = batch.num_keys * sizeof(InsertStatus);
    size_t hops_size = batch.num_keys * sizeof(uint32_t);

    cudaMalloc(&d_keys, keys_size);
    cudaMalloc(&d_values, keys_size);
    cudaMalloc(&d_statuses, statuses_size);
    if (batch.h_hops) cudaMalloc(&d_hops, hops_size);

    cudaMemcpyAsync(d_keys, batch.h_keys, keys_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_values, batch.h_values, keys_size, cudaMemcpyHostToDevice, stream);

    uint32_t threads_per_block = 256;
    uint32_t keys_per_block = threads_per_block / 32;
    uint32_t num_blocks = (batch.num_keys + keys_per_block - 1) / keys_per_block;

    warp_insert_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        table, d_stash, d_keys, d_values, d_statuses, d_hops, batch.num_keys
    );

    cudaMemcpyAsync(batch.h_statuses, d_statuses, statuses_size, cudaMemcpyDeviceToHost, stream);
    if (batch.h_hops && d_hops) {
        cudaMemcpyAsync(batch.h_hops, d_hops, hops_size, cudaMemcpyDeviceToHost, stream);
    }

    if (stream == nullptr) cudaDeviceSynchronize();

    cudaFree(d_keys);
    cudaFree(d_values);
    cudaFree(d_statuses);
    if (d_hops) cudaFree(d_hops);
}

}  // namespace warpkv
